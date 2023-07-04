#ifndef TMVA_RBatchLoader
#define TMVA_RBatchLoader

#include <iostream>
#include <vector>
#include <memory>

// Imports for threading
#include <queue>
#include <mutex>
#include <condition_variable>

#include "TMVA/RTensor.hxx"
#include "TMVA/Tools.h"
#include "TRandom3.h"

namespace TMVA {
namespace Experimental {

class RBatchLoader {
private:
   const size_t fBatchSize, fNumColumns, fMaxBatches;

   bool fIsActive = false;
   TMVA::RandomGenerator<TRandom3> fRng;

   // filled batch elements
   std::mutex fBatchLock;
   std::condition_variable fBatchCondition;

   std::queue<std::unique_ptr<TMVA::Experimental::RTensor<float>>> fTrainingBatchQueue;
   std::vector<std::unique_ptr<TMVA::Experimental::RTensor<float>>> fValidationBatches;

   std::unique_ptr<TMVA::Experimental::RTensor<float>> fCurrentBatch;
   size_t fValidationIdx = 0;

public:
   /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   // Constructors
   /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   RBatchLoader(const size_t batchSize, const size_t numColumns, const size_t maxBatches)
      : fBatchSize(batchSize), fNumColumns(numColumns), fMaxBatches(maxBatches)
   {
      fRng = TMVA::RandomGenerator<TRandom3>(0);
   }

   ~RBatchLoader() { DeActivate(); }

public:
   /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   // Batch functions
   /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

   // return a batch of data
   TMVA::Experimental::RTensor<float> *GetTrainBatch()
   {
      std::unique_lock<std::mutex> lock(fBatchLock);
      fBatchCondition.wait(lock, [this]() { return !fTrainingBatchQueue.empty() || !fIsActive; });

      if (fTrainingBatchQueue.empty())
         return nullptr;

      fCurrentBatch = std::move(fTrainingBatchQueue.front());
      fTrainingBatchQueue.pop();

      fBatchCondition.notify_all();

      return fCurrentBatch.get();
   }

   // return a batch of data
   TMVA::Experimental::RTensor<float> *GetValidationBatch() { return fValidationBatches[fValidationIdx++].get(); }

   bool HasTrainData()
   {
      {
         std::unique_lock<std::mutex> lock(fBatchLock);
         if (!fTrainingBatchQueue.empty() || fIsActive)
            return true;
      }

      return false;
   }

   bool HasValidationData()
   {
      std::unique_lock<std::mutex> lock(fBatchLock);
      return fValidationIdx < fValidationBatches.size();
   }

   // Activate the batchloader so it will accept chunks to batch
   void Activate()
   {
      {
         std::lock_guard<std::mutex> lock(fBatchLock);
         fIsActive = true;
      }
      fBatchCondition.notify_all();
   }

   // DeActivate the batchloader. This means that no more batches are created.
   // Batches can still be gotten if available
   void DeActivate()
   {
      {
         std::lock_guard<std::mutex> lock(fBatchLock);
         fIsActive = false;
      }
      fBatchCondition.notify_all();
   }

   /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   // Getters and Setters
   /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

   // Create a batch filled with the events on the given idx
   std::unique_ptr<TMVA::Experimental::RTensor<float>>
   CreateBatch(const TMVA::Experimental::RTensor<float> &chunkTensor, const std::vector<size_t> idx)
   {
      auto batch = std::unique_ptr<TMVA::Experimental::RTensor<float>>(
         new TMVA::Experimental::RTensor<float>({fBatchSize, fNumColumns}));
      for (size_t i = 0; i < fBatchSize; i++) {
         std::copy(chunkTensor.GetData() + (idx[i] * fNumColumns), chunkTensor.GetData() + ((idx[i] + 1) * fNumColumns),
                   batch->GetData() + i * fNumColumns);
      }

      return batch;
   }

   // Batch the given chunk into batches based on the given event indices
   // Batches are added to the training queue of batches
   // The eventIndices can be shuffled to ensure random order for each epoch
   void CreateTrainingBatches(const TMVA::Experimental::RTensor<float> &chunkTensor, std::vector<size_t> eventIndices,
                              const bool shuffle = true)
   {
      // Wait until less than a full chunk of batches are in the queue before loading splitting the next chunk into
      // batches
      {
         std::unique_lock<std::mutex> lock(fBatchLock);
         fBatchCondition.wait(lock, [this]() { return (fTrainingBatchQueue.size() < fMaxBatches) || !fIsActive; });
         if (!fIsActive)
            return;
      }

      if (shuffle)
         std::shuffle(eventIndices.begin(), eventIndices.end(), fRng); // Shuffle the order of idx

      std::vector<std::unique_ptr<TMVA::Experimental::RTensor<float>>> batches;

      // Create tasks of fBatchSize untill all idx are used
      for (size_t start = 0; (start + fBatchSize) <= eventIndices.size(); start += fBatchSize) {

         // Grab the first fBatchSize indices from the
         std::vector<size_t> idx;
         for (size_t i = start; i < (start + fBatchSize); i++) {
            idx.push_back(eventIndices[i]);
         }

         // Fill a batch
         batches.emplace_back(CreateBatch(chunkTensor, idx));
      }

      {
         std::unique_lock<std::mutex> lock(fBatchLock);
         for (size_t i = 0; i < batches.size(); i++) {
            fTrainingBatchQueue.push(std::move(batches[i]));
         }
      }

      fBatchCondition.notify_one();
   }

   // Batch the given chunk into batches based on the given event indices
   // Batches are added to the vector of validation batches
   void CreateValidationBatches(const TMVA::Experimental::RTensor<float> &chunkTensor,
                                const std::vector<size_t> eventIndices)
   {
      // Create tasks of fBatchSize untill all idx are used
      for (size_t start = 0; (start + fBatchSize) <= eventIndices.size(); start += fBatchSize) {

         std::vector<size_t> idx;

         for (size_t i = start; i < (start + fBatchSize); i++) {
            idx.push_back(eventIndices[i]);
         }

         {
            std::unique_lock<std::mutex> lock(fBatchLock);
            fValidationBatches.emplace_back(CreateBatch(chunkTensor, idx));
         }
      }
   }

   void StartValidation()
   {
      std::unique_lock<std::mutex> lock(fBatchLock);
      fValidationIdx = 0;
   }
};

} // namespace Experimental
} // namespace TMVA

#endif // TMVA_RBatchLoader