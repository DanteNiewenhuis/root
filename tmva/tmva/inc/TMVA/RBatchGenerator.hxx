#ifndef TMVA_BATCHGENERATOR
#define TMVA_BATCHGENERATOR

#include <iostream>
#include <vector>
#include <thread>
#include <memory>
#include <cmath>
#include <mutex>

#include "TMVA/RTensor.hxx"
#include "ROOT/RDF/RDatasetSpec.hxx"
#include "TMVA/RChunkLoader.hxx"
#include "TMVA/RBatchLoader.hxx"
#include "TMVA/Tools.h"
#include "TRandom3.h"

namespace TMVA {
namespace Experimental {

template <typename... Args>
class RBatchGenerator {
private:
   TMVA::RandomGenerator<TRandom3> fRng;

   std::vector<std::string> fCols, fFilters;
   size_t fNumColumns, fChunkSize, fMaxChunks, fBatchSize, fCurrentRow = 0, fNumEntries, fMaxBatches;

   std::string fFileName, fTreeName;

   std::unique_ptr<TMVA::Experimental::RBatchLoader> fBatchLoader;

   std::unique_ptr<std::thread> fLoadingThread;

   bool fEoF = false, fUseWholeFile = true;
   double fValidationSplit;

   std::unique_ptr<TMVA::Experimental::RTensor<float>> fChunkTensor;
   std::unique_ptr<TMVA::Experimental::RTensor<float>> fCurrentBatch;

   std::vector<std::vector<size_t>> fTrainingIdxs;
   std::vector<std::vector<size_t>> fValidationIdxs;

   bool fShuffle = true, fActivated = false;

   std::vector<size_t> fVecSizes;
   float fVecPadding;

   std::mutex fDataFrameLock;

   ////////////////////////////////////////////////////////////////////////////////////////////////////////
   /// Functions
   ////////////////////////////////////////////////////////////////////////////////////////////////////////

   // Load fChunkSize rows of the given RDataFrame into a RTensor.
   // After, the chunk of data is split into batches of data.
   void LoadChunk(size_t current_chunk)
   {

      const std::lock_guard<std::mutex> lock(fDataFrameLock);
      TMVA::Experimental::RChunkLoader<Args...> func((*fChunkTensor), fVecSizes, fVecPadding);

      // Create TDataFrame of the chunk
      // Use RDatasetSpec to start reading at the current row
      long long start_l = fCurrentRow;
      ROOT::RDF::Experimental::RDatasetSpec x_spec =
         ROOT::RDF::Experimental::RDatasetSpec()
            .AddSample({"", fTreeName, fFileName})
            .WithGlobalRange({start_l, std::numeric_limits<Long64_t>::max()});
      ROOT::RDataFrame x_rdf(x_spec);

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
         passed_events = (myReport.end() - 1)->GetPass();
      }

      // load events if no filters are given
      else {
         // add range
         auto x_ranged = x_rdf.Range(fChunkSize);
         auto myCount = x_ranged.Count();

         // load data
         x_ranged.Foreach(func, fCols);

         // get loading info
         processed_events = myCount.GetValue();
         passed_events = myCount.GetValue();
      }

      fCurrentRow += processed_events;

      // Create batches for the current_chunk.
      // First get the correct idices to use, then turn them into batches
      // Validation batches only have to be made in the first epoch
      if (fTrainingIdxs.size() > current_chunk) {
         fBatchLoader->CreateTrainingBatches(*fChunkTensor, fTrainingIdxs[current_chunk], fShuffle);
      } else {
         createIdxs(current_chunk, processed_events);
         fBatchLoader->CreateTrainingBatches(*fChunkTensor, fTrainingIdxs[current_chunk], fShuffle);
         fBatchLoader->CreateValidationBatches(*fChunkTensor, fValidationIdxs[current_chunk]);
      }
   }

   // Split the event of the current chunk into validation and training events
   void createIdxs(size_t current_chunk, size_t processed_events)
   {
      std::vector<size_t> row_order = std::vector<size_t>(processed_events);

      std::iota(row_order.begin(), row_order.end(), 0);

      if (fShuffle) {
         std::shuffle(row_order.begin(), row_order.end(), fRng);
      }

      size_t num_validation = ceil(processed_events * fValidationSplit);

      std::vector<size_t> valid_idx({row_order.begin(), row_order.begin() + num_validation});
      std::vector<size_t> train_idx({row_order.begin() + num_validation, row_order.end()});

      fTrainingIdxs.push_back(train_idx);
      fValidationIdxs.push_back(valid_idx);
   }

public:
   RBatchGenerator(std::string fileName, std::string treeName, size_t chunkSize, size_t batchSize,
                   std::vector<std::string> cols, std::vector<std::string> filters, std::vector<size_t> vecSizes = {},
                   float vecPadding = 0.0, double validationSplit = 0.0, size_t maxChunks = 0, size_t numColumns = 0,
                   bool shuffle = true)
      : fFileName(fileName),
        fTreeName(treeName),
        fChunkSize(chunkSize),
        fBatchSize(batchSize),
        fCols(cols),
        fFilters(filters),
        fVecSizes(vecSizes),
        fVecPadding(vecPadding),
        fValidationSplit(validationSplit),
        fMaxChunks(maxChunks),
        fNumColumns(numColumns),
        fShuffle(shuffle)
   {
      if (fMaxChunks > 0) {
         fUseWholeFile = false;
      }
      if (fNumColumns == 0) {
         fNumColumns = fCols.size();
      }

      fMaxBatches = ceil((fChunkSize / fBatchSize) * (1 - fValidationSplit));

      // get the number of fNumEntries in the dataframe
      TFile *f = TFile::Open(fFileName.c_str());
      TTree *t = f->Get<TTree>(fTreeName.c_str());
      fNumEntries = t->GetEntries();
      std::cout << "RBatchGenerator => found " << fNumEntries << " fNumEntries in file." << std::endl;

      fBatchLoader = std::make_unique<TMVA::Experimental::RBatchLoader>(fBatchSize, fNumColumns, fMaxBatches);
      fRng = TMVA::RandomGenerator<TRandom3>(0);

      // Create tensor to load the chunk into
      fChunkTensor = std::unique_ptr<TMVA::Experimental::RTensor<float>>(
         new TMVA::Experimental::RTensor<float>({fChunkSize, fNumColumns}));
   }

   ~RBatchGenerator() { DeActivate(); }

   void DeActivate()
   {
      fBatchLoader->DeActivate();
      fActivated = false;

      if (fLoadingThread) {
         fLoadingThread->join();
         fLoadingThread = nullptr;
      }
   }

   void Activate()
   {
      if (fActivated) {
         DeActivate();
      }

      fCurrentRow = 0;
      fActivated = true;

      fBatchLoader->Activate();
      fLoadingThread = std::make_unique<std::thread>(&RBatchGenerator::LoadChunks, this);
   }

   // Returns the next batch of data if available.
   // Returns empty RTensor otherwise.
   TMVA::Experimental::RTensor<float> *GetTrainBatch()
   {
      // Get next batch if available
      if (fBatchLoader->HasTrainData()) {
         fCurrentBatch = fBatchLoader->GetTrainBatch();
         return fCurrentBatch.get();
      }

      return nullptr;
   }

   // Returns the next batch of data if available.
   // Returns empty RTensor otherwise.
   TMVA::Experimental::RTensor<float> *GetValidationBatch()
   {
      // Get next batch if available
      if (fBatchLoader->HasValidationData()) {
         return fBatchLoader->GetValidationBatch();
      }

      // return empty batch if all events have been used
      return nullptr;
   }

   bool HasTrainData()
   {
      if (!fBatchLoader->HasTrainData() && fEoF) {
         return false;
      }

      return true;
   }

   bool HasValidationData()
   {
      if (!fBatchLoader->HasValidationData()) {
         return false;
      }

      return true;
   }

   void LoadChunks()
   {
      fEoF = false;

      // Load chunks untill the end of the file is reached.
      // Stop loading if a maximum number of chunks is provided
      for (size_t i = 0; ((i < fMaxChunks) || fUseWholeFile) && fActivated; i++) {
         LoadChunk(i);
         if (fCurrentRow >= fNumEntries) {
            break;
         }
      }

      fEoF = true;
      fBatchLoader->DeActivate();
   }

   void StartValidation() { fBatchLoader->StartValidation(); }
};

} // namespace Experimental
} // namespace TMVA

#endif // TMVA_BATCHGENERATOR