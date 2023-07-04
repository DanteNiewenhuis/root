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
#include "TROOT.h"

namespace TMVA {
namespace Experimental {

template <typename... Args>
class RBatchGenerator {
private:
   TMVA::RandomGenerator<TRandom3> fRng;

   const std::string fFileName, fTreeName;
   const std::vector<std::string> fCols, fFilters;

   const size_t fChunkSize, fMaxChunks, fBatchSize;
   size_t fNumColumns;
   size_t fCurrentRow = 0, fMaxBatches, fNumEntries;

   std::unique_ptr<TMVA::Experimental::RChunkLoader<Args...>> fChunkLoader;
   std::unique_ptr<TMVA::Experimental::RBatchLoader> fBatchLoader;

   std::unique_ptr<std::thread> fLoadingThread;

   const bool fUseWholeFile = true;

   std::unique_ptr<TMVA::Experimental::RTensor<float>> fChunkTensor;
   std::unique_ptr<TMVA::Experimental::RTensor<float>> fCurrentBatch;

   std::vector<std::vector<size_t>> fTrainingIdxs;
   std::vector<std::vector<size_t>> fValidationIdxs;

   // filled batch elements
   std::mutex fIsActiveLock;

   bool fShuffle = true, fIsActive = false;

   const std::vector<size_t> fVecSizes;
   const float fVecPadding, fValidationSplit;

   ////////////////////////////////////////////////////////////////////////////////////////////////////////
   /// Functions
   ////////////////////////////////////////////////////////////////////////////////////////////////////////

public:
   RBatchGenerator(const std::string treeName, const std::string fileName, const size_t chunkSize,
                   const size_t batchSize, const std::vector<std::string> cols,
                   const std::vector<std::string> filters = {}, const std::vector<size_t> vecSizes = {},
                   const float vecPadding = 0.0, const float validationSplit = 0.0, const size_t maxChunks = 0,
                   const size_t numColumns = 0, bool shuffle = true)
      : fTreeName(treeName),
        fFileName(fileName),
        fChunkSize(chunkSize),
        fBatchSize(batchSize),
        fCols(cols),
        fFilters(filters),
        fVecSizes(vecSizes),
        fVecPadding(vecPadding),
        fValidationSplit(validationSplit),
        fMaxChunks(maxChunks),
        fNumColumns((numColumns != 0) ? numColumns : cols.size()),
        fShuffle(shuffle),
        fUseWholeFile(maxChunks == 0)
   {
      fMaxBatches = ceil((fChunkSize / fBatchSize) * (1 - fValidationSplit));

      // get the number of fNumEntries in the dataframe
      TFile *f = TFile::Open(fFileName.c_str());
      TTree *t = f->Get<TTree>(fTreeName.c_str());
      fNumEntries = t->GetEntries();
      std::cout << "RBatchGenerator => found " << fNumEntries << " fNumEntries in file." << std::endl;

      fChunkLoader = std::make_unique<TMVA::Experimental::RChunkLoader<Args...>>(
         fTreeName, fFileName, fChunkSize, fCols, fFilters, fVecSizes, fVecPadding);
      fBatchLoader = std::make_unique<TMVA::Experimental::RBatchLoader>(fBatchSize, fNumColumns, fMaxBatches);

      fRng = TMVA::RandomGenerator<TRandom3>(0);

      // Create tensor to load the chunk into
      fChunkTensor = std::unique_ptr<TMVA::Experimental::RTensor<float>>(
         new TMVA::Experimental::RTensor<float>({fChunkSize, fNumColumns}));
   }

   ~RBatchGenerator()
   {
      std::cout << "RBatchGenerator::Distructor => init" << std::endl;

      DeActivate();
   }

   void DeActivate()
   {
      std::cout << "RBatchGenerator::DeActivate => init" << std::endl;

      {
         std::lock_guard<std::mutex> lock(fIsActiveLock);
         std::cout << "RBatchGenerator::DeActivate => locked thread" << std::endl;
         fIsActive = false;
      }

      fBatchLoader->DeActivate();

      std::cout << "RBatchGenerator::DeActivate => unlocked" << std::endl;

      if (fLoadingThread) {
         if (fLoadingThread->joinable()) {
            std::cout << "RBatchGenerator::DeActivate => Joining Thread " << fLoadingThread.get() << std::endl;
            fLoadingThread->join();
            std::cout << "RBatchGenerator::DeActivate => Joined Thread" << std::endl;
         }
      }
   }

   void Activate()
   {
      if (fIsActive) {
         return;
      }

      {
         std::lock_guard<std::mutex> lock(fIsActiveLock);
         std::cout << "RBatchGenerator::Activate => locked thread" << std::endl;
         fIsActive = true;
      }

      fCurrentRow = 0;
      fBatchLoader->Activate();
      fLoadingThread = std::make_unique<std::thread>(&RBatchGenerator::LoadChunks, this);

      std::cout << "RBatchGenerator::Activate => done" << std::endl;
   }

   // Returns the next batch of data if available.
   // Returns empty RTensor otherwise.
   TMVA::Experimental::RTensor<float> *GetTrainBatch()
   {
      // Get next batch if available
      if (fBatchLoader->HasTrainData()) {
         return fBatchLoader->GetTrainBatch();
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

   bool HasTrainData() { return fBatchLoader->HasTrainData(); }

   bool HasValidationData() { return fBatchLoader->HasValidationData(); }

   void LoadChunks()
   {
      // Load chunks untill the end of the file is reached.
      // Stop loading if a maximum number of chunks is provided

      std::cout << "RBatchGenerator::LoadChunks => init: " << fCurrentRow << " entries: " << fNumEntries << std::endl;
      size_t processed_events;
      for (size_t current_chunk = 0; ((current_chunk < fMaxChunks) || fUseWholeFile) && (fCurrentRow < fNumEntries);
           current_chunk++) {

         // Make sure the loading has not been shut off
         {
            std::lock_guard<std::mutex> lock(fIsActiveLock);
            std::cout << "RBatchGenerator::LoadChunks => locked thread: " << current_chunk << " active?: " << fIsActive
                      << std::endl;

            if (!fIsActive) {
               return;
            }
         }

         std::cout << "RBatchGenerator::LoadChunks => unlocked thread: " << current_chunk << std::endl;
         std::cout << "RBatchGenerator::LoadChunks => loading chunk: " << current_chunk << std::endl;
         processed_events = fChunkLoader->LoadChunk(*fChunkTensor, fCurrentRow);
         fCurrentRow += processed_events;

         std::cout << "RBatchGenerator::LoadChunks => processed_events: " << processed_events << std::endl;
         std::cout << "RBatchGenerator::LoadChunks => batching chunk: " << current_chunk << std::endl;
         CreateBatches(current_chunk, processed_events);
      }

      fBatchLoader->DeActivate();
   }

   // Create batches for the current_chunk.
   void CreateBatches(size_t current_chunk, size_t processed_events)
   {

      // Check if the indices in this chunk where already split in train and validations
      if (fTrainingIdxs.size() > current_chunk) {
         fBatchLoader->CreateTrainingBatches(*fChunkTensor, fTrainingIdxs[current_chunk], fShuffle);
      } else {
         // Create the Validation batches if this is not the first epoch
         createIdxs(current_chunk, processed_events);
         fBatchLoader->CreateTrainingBatches(*fChunkTensor, fTrainingIdxs[current_chunk], fShuffle);
         fBatchLoader->CreateValidationBatches(*fChunkTensor, fValidationIdxs[current_chunk]);
      }
   }

   // Split the event of the current chunk into validation and training events
   void createIdxs(size_t current_chunk, size_t processed_events)
   {
      // Create a vector of number 1..processed_events
      std::vector<size_t> row_order = std::vector<size_t>(processed_events);
      std::iota(row_order.begin(), row_order.end(), 0);

      if (fShuffle) {
         std::shuffle(row_order.begin(), row_order.end(), fRng);
      }

      // calculate the number of events used for validation
      size_t num_validation = ceil(processed_events * fValidationSplit);

      // Devide the vector into training and validation
      std::vector<size_t> valid_idx({row_order.begin(), row_order.begin() + num_validation});
      std::vector<size_t> train_idx({row_order.begin() + num_validation, row_order.end()});

      fTrainingIdxs.push_back(train_idx);
      fValidationIdxs.push_back(valid_idx);
   }

   void StartValidation() { fBatchLoader->StartValidation(); }
   bool IsActive() { return fIsActive; }
};

} // namespace Experimental
} // namespace TMVA

#endif // TMVA_BATCHGENERATOR