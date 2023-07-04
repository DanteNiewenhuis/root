#ifndef TMVA_CHUNKLOADER
#define TMVA_CHUNKLOADER

#include <iostream>
#include <vector>

#include "TMVA/RTensor.hxx"
#include "ROOT/RDataFrame.hxx"
#include "ROOT/RVec.hxx"

#include "ROOT/RLogger.hxx"

namespace TMVA {
namespace Experimental {

// RChunkLoader class used to load content of a RDataFrame onto a RTensor.
template <typename First, typename... Rest>
class RChunkLoaderFunctor {

private:
   size_t fOffset = 0, fVecSizeIdx = 0;
   const std::vector<size_t> fMaxVecSizes;

   const float fVecPadding;

   TMVA::Experimental::RTensor<float> &fChunkTensor;

   /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   // Value assigning
   /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

   // Load the final given value into fChunkTensor
   // Add a label to the end of the row if given
   template <typename First_T>
   void AssignToTensor(First_T first)
   {
      fChunkTensor.GetData()[fOffset++] = first;
   }
   // Vector version of the previous function
   template <typename VecType>
   void AssignToTensor(ROOT::RVec<VecType> first)
   {
      AssignVector(first);
   }

   // Recursively loop through the given values, and load them onto the fChunkTensor
   template <typename First_T, typename... Rest_T>
   void AssignToTensor(First_T first, Rest_T... rest)
   {
      fChunkTensor.GetData()[fOffset++] = first;

      AssignToTensor(std::forward<Rest_T>(rest)...);
   }

   // Vector version of the previous function
   template <typename VecType, typename... Rest_T>
   void AssignToTensor(ROOT::RVec<VecType> first, Rest_T... rest)
   {
      AssignVector(first);

      AssignToTensor(std::forward<Rest_T>(rest)...);
   }

   // Loop through the values of a given vector and load them into the RTensor
   // Note: the given vec_size does not have to be the same size as the given vector
   //       If the size is bigger than the given vector, zeros are used as padding.
   //       If the size is smaller, the remaining values are ignored.
   template <typename VecType>
   void AssignVector(ROOT::RVec<VecType> vec)
   {
      size_t max_vec_size = fMaxVecSizes[fVecSizeIdx++];
      size_t vec_size = vec.size();

      for (size_t i = 0; i < max_vec_size; i++) {
         if (i < vec_size) {
            // TODO: add better type conversion
            fChunkTensor.GetData()[fOffset++] = vec[i];
         } else {
            fChunkTensor.GetData()[fOffset++] = fVecPadding;
         }
      }
   }

public:
   /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   // Constructor
   /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   RChunkLoaderFunctor(TMVA::Experimental::RTensor<float> &chunkTensor,
                       const std::vector<size_t> maxVecSizes = std::vector<size_t>(), const float vecPadding = 0.0)
      : fChunkTensor(chunkTensor), fMaxVecSizes(maxVecSizes), fVecPadding(vecPadding)
   {
   }

   void operator()(First first, Rest... rest)
   {
      fVecSizeIdx = 0;
      AssignToTensor(std::forward<First>(first), std::forward<Rest>(rest)...);
   }
};

template <typename... Args>
class RChunkLoader {

private:
   const std::string fTreeName, fFileName;
   const size_t fChunkSize, fNumColumns;

   const std::vector<std::string> fCols, fFilters;

   const std::vector<size_t> fVecSizes;
   const size_t fVecPadding;

public:
   RChunkLoader(const std::string treeName, const std::string fileName, const size_t chunkSize,
                const std::vector<std::string> cols, const std::vector<std::string> filters = {},
                const std::vector<size_t> vecSizes = {}, const float vecPadding = 0.0)
      : fTreeName(treeName),
        fFileName(fileName),
        fChunkSize(chunkSize),
        fCols(cols),
        fFilters(filters),
        fVecSizes(vecSizes),
        fVecPadding(vecPadding),
        fNumColumns(cols.size())
   {
   }

   size_t LoadChunk(TMVA::Experimental::RTensor<float> &chunk_tensor, const size_t currentRow)
   {
      std::cout << "RChunkLoader::LoadChunk => init: " << currentRow << std::endl;
      // auto verbosity = ROOT::Experimental::RLogScopedVerbosity(ROOT::Detail::RDF::RDFLogChannel(),
      //                                                          ROOT::Experimental::ELogLevel::kInfo);
      RChunkLoaderFunctor<Args...> func(chunk_tensor, fVecSizes, fVecPadding);

      // Create TDataFrame of the chunk
      // Use RDatasetSpec to start reading at the current row
      std::cout << "RChunkLoader::LoadChunk => Creating dataset spec: " << currentRow << std::endl;
      long long start_l = currentRow;
      ROOT::RDF::Experimental::RDatasetSpec x_spec =
         ROOT::RDF::Experimental::RDatasetSpec()
            .AddSample({"", fTreeName, fFileName})
            .WithGlobalRange({start_l, std::numeric_limits<Long64_t>::max()});

      std::cout << "RChunkLoader::LoadChunk => create RDF: " << currentRow << std::endl;
      ROOT::RDataFrame x_rdf(x_spec);

      // ROOT::RDataFrame x_rdf = ROOT::RDataFrame(fTreeName, fFileName);

      std::cout << "RChunkLoader::LoadChunk => Created RDF: " << currentRow << std::endl;
      size_t processed_events, passed_events;

      // Load events if filters are given
      if (fFilters.size() > 0) {

         // Add the given filters to the RDataFrame
         auto x_filter = x_rdf.Filter(fFilters[0], "F1");
         for (auto i = 1; i < fFilters.size(); i++) {
            auto name = "F" + std::to_string(i);
            x_filter = x_filter.Filter(fFilters[i], name);
         }

         // add range to the DataFrame
         auto x_ranged = x_filter.Range(fChunkSize);
         auto myReport = x_ranged.Report();

         // load data
         x_ranged.Foreach(func, fCols);

         // Use the report to gather the number of events processed and passed.
         // passed_events is used to determine the starting event of the next chunk
         // processed_events is used to determine if the end of the database is reached.
         processed_events = myReport.begin()->GetAll();

         // TODO: return passed_events
         passed_events = (myReport.end() - 1)->GetPass();
      }

      // load events if no filters are given
      else {
         // add range
         auto x_ranged = x_rdf.Range(fChunkSize);
         // auto x_ranged = x_rdf.Range(currentRow, currentRow + fChunkSize);
         auto myCount = x_ranged.Count();

         std::cout << "RChunkLoader::LoadChunk => start Foreach: " << currentRow << std::endl;
         // load data
         x_ranged.Foreach(func, fCols);

         // get loading info
         processed_events = myCount.GetValue();
         passed_events = myCount.GetValue();
      }

      std::cout << "RChunkLoader::LoadChunk => finished: " << currentRow << std::endl;
      return processed_events;
   }
};

} // namespace Experimental
} // namespace TMVA
#endif // TMVA_CHUNKLOADER