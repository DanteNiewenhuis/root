#ifndef TMVA_BATCHGENERATOR
#define TMVA_BATCHGENERATOR

#include <iostream>
#include <vector>
#include <thread>
#include <memory>

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
   size_t fNumColumns, fChunkSize, fMaxChunks, fBatchSize, fCurrentRow = 0, fNumEntries;

   std::string fFileName, fTreeName;

   std::unique_ptr<TMVA::Experimental::RBatchLoader> fBatchLoader;

   std::unique_ptr<std::thread> fLoadingThread;

   bool fEoF = false, fUseWholeFile = true;
   double fValidationSplit;

   std::unique_ptr<TMVA::Experimental::RTensor<float>> fChunkTensor;
   std::unique_ptr<TMVA::Experimental::RTensor<float>> fCurrentBatch;

   std::vector<std::vector<size_t>> fTrainingIdxs;
   std::vector<std::vector<size_t>> fValidationIdxs;

   bool fShuffle = true;

   std::vector<size_t> fVecSizes;

   ////////////////////////////////////////////////////////////////////////////////////////////////////////
   /// Functions
   ////////////////////////////////////////////////////////////////////////////////////////////////////////

   // Load fChunkSize rows of the given RDataFrame into a RTensor.
   // After, the chunk of data is split into batches of data.
   void LoadChunk(size_t current_chunk)
   {
      TMVA::Experimental::RChunkLoader<Args...> func((*fChunkTensor), fVecSizes);

      // Create DataFrame
      long long start_l = fCurrentRow;
      long long end_l = start_l + fChunkSize;
      ROOT::RDF::Experimental::RDatasetSpec x_spec =
         ROOT::RDF::Experimental::RDatasetSpec()
            .AddSample({"", fTreeName, fFileName})
            .WithGlobalRange({start_l, std::numeric_limits<Long64_t>::max()});
      ROOT::RDataFrame x_rdf(x_spec);

      size_t progressed_events, passed_events;

      // add fFilters if given
      if (fFilters.size() > 0) {
         auto x_filter = x_rdf.Filter(fFilters[0], "F1");

         for (auto i = 1; i < fFilters.size(); i++) {
            auto name = "F" + std::to_string(i);
            x_filter = x_filter.Filter(fFilters[i], name);
         }

         // add range
         auto x_ranged = x_filter.Range(fChunkSize);
         auto myReport = x_ranged.Report();

         // load data
         x_ranged.Foreach(func, fCols);

         // get the loading info
         progressed_events = myReport.begin()->GetAll();
         passed_events = (myReport.end() - 1)->GetPass();
      }

      // no fFilters given
      else {
         // add range
         auto x_ranged = x_rdf.Range(fChunkSize);
         auto myCount = x_ranged.Count();

         // load data
         x_ranged.Foreach(func, fCols);

         // get loading info
         progressed_events = myCount.GetValue();
         passed_events = myCount.GetValue();
      }

      // std::cout << "RBatchGenerator::Init => tensor: " << fChunkTensor << std::endl;

      fCurrentRow += progressed_events;

      // Create batches for the current_chunk.
      // First get the correct idices to use, then turn them into batches
      // Validation batches only have to be made in the first epoch
      if (fTrainingIdxs.size() > current_chunk) {
         fBatchLoader->CreateTrainingBatches(*fChunkTensor, fTrainingIdxs[current_chunk], fShuffle);
      } else {
         createIdxs(current_chunk, progressed_events);
         fBatchLoader->CreateTrainingBatches(*fChunkTensor, fTrainingIdxs[current_chunk], fShuffle);
         fBatchLoader->CreateValidationBatches(*fChunkTensor, fValidationIdxs[current_chunk]);
      }
   }
   void createIdxs(size_t current_chunk, size_t progressed_events)
   {
      std::vector<size_t> row_order = std::vector<size_t>(progressed_events);

      std::iota(row_order.begin(), row_order.end(), 0);

      if (fShuffle) {
         std::shuffle(row_order.begin(), row_order.end(), fRng);
      }

      size_t num_validation = progressed_events * fValidationSplit;

      std::vector<size_t> valid_idx({row_order.begin(), row_order.begin() + num_validation});
      std::vector<size_t> train_idx({row_order.begin() + num_validation, row_order.end()});

      fTrainingIdxs.push_back(train_idx);
      fValidationIdxs.push_back(valid_idx);
   }

public:
   RBatchGenerator(std::string fileName, std::string treeName, std::vector<std::string> cols,
                   std::vector<std::string> filters, size_t chunkSize, size_t batchSize,
                   std::vector<size_t> vecSizes = {}, double validationSplit = 0.0, size_t maxChunks = 0,
                   size_t numColumns = 0, bool shuffle = true)
      : fFileName(fileName),
        fTreeName(treeName),
        fCols(cols),
        fFilters(filters),
        fChunkSize(chunkSize),
        fBatchSize(batchSize),
        fVecSizes(vecSizes),
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

      // get the number of fNumEntries in the dataframe
      TFile *f = TFile::Open(fFileName.c_str());
      TTree *t = f->Get<TTree>(fTreeName.c_str());
      fNumEntries = t->GetEntries();
      std::cout << "RBatchGenerator => found " << fNumEntries << " fNumEntries in file." << std::endl;

      fBatchLoader = std::make_unique<TMVA::Experimental::RBatchLoader>(fBatchSize, fNumColumns);
      fRng = TMVA::RandomGenerator<TRandom3>(0);

      // Create tensor to load the chunk into
      fChunkTensor = std::unique_ptr<TMVA::Experimental::RTensor<float>>(
         new TMVA::Experimental::RTensor<float>({fChunkSize, fNumColumns}));
   }

   ~RBatchGenerator() { StopLoading(); }

   void StopLoading()
   {
      if (fLoadingThread) {
         fLoadingThread->join();
         fLoadingThread = nullptr;
      }
   }

   void Init()
   {
      std::cout << "Cpp::RBatchGenerator => Init" << std::endl;

      fCurrentRow = 0;
      fBatchLoader->Activate();
      fLoadingThread = std::make_unique<std::thread>(&RBatchGenerator::LoadChunks, this);
   }

   // Returns the next batch of data if available.
   // Returns empty RTensor otherwise.
   TMVA::Experimental::RTensor<float> *GetTrainBatch()
   {

      std::cout << "Cpp::RBatchGenerator::GetTrainBatch => Init" << std::endl;

      // Get next batch if available
      if (fBatchLoader->HasTrainData()) {
         fCurrentBatch = fBatchLoader->GetTrainBatch();
         return fCurrentBatch.get();
      }

      // return empty batch if all events have been used
      return new TMVA::Experimental::RTensor<float>({0, 0});
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
      return new TMVA::Experimental::RTensor<float>({0, 0});
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
      for (size_t i = 0; ((i < fMaxChunks) || fUseWholeFile); i++) {
         LoadChunk(i);
         if (fCurrentRow >= fNumEntries) {
            break;
         }
      }

      fBatchLoader->DeActivate();
      fEoF = true;
   }

   void StartValidation() { fBatchLoader->StartValidation(); }
};

} // namespace Experimental
} // namespace TMVA

#endif // TMVA_BATCHGENERATOR