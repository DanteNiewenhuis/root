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

   bool accept_tasks = false;
   TMVA::RandomGenerator<TRandom3> fRng;

   // filled batch elements
   std::mutex fBatchLock;
   std::condition_variable fBatchCondition;

   std::queue<std::unique_ptr<TMVA::Experimental::RTensor<float>>> fTrainingBatchQueue;
   std::vector<std::unique_ptr<TMVA::Experimental::RTensor<float>>> fValidationBatches;
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

   ~RBatchLoader()
   {
      std::cout << "Cpp::RBatchLoader => Deconstructor" << std::endl;
      DeActivate();
   }

public:
   /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   // Batch functions
   /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

   // return a batch of data
   std::unique_ptr<TMVA::Experimental::RTensor<float>> GetTrainBatch()
   {

      std::unique_lock<std::mutex> lock(fBatchLock);
      fBatchCondition.wait(lock, [this]() { return !fTrainingBatchQueue.empty() || !accept_tasks; });

      if (fTrainingBatchQueue.empty()) {
         return std::unique_ptr<TMVA::Experimental::RTensor<float>>();
      }

      std::unique_ptr<TMVA::Experimental::RTensor<float>> front = std::move(fTrainingBatchQueue.front());
      fTrainingBatchQueue.pop();

      fBatchCondition.notify_all();
      return front;
   }

   // return a batch of data
   TMVA::Experimental::RTensor<float> *GetValidationBatch() { return fValidationBatches[fValidationIdx++].get(); }

   bool HasTrainData()
   {
      std::unique_lock<std::mutex> lock(fBatchLock);
      if (!fTrainingBatchQueue.empty() || accept_tasks)
         return true;
      lock.unlock();

      return false;
   }

   bool HasValidationData() { return fValidationIdx < fValidationBatches.size(); }

   // Activate the threads again to accept new tasks
   void Activate()
   {
      std::cout << "Cpp::RBatchLoader => Activate" << std::endl;
      accept_tasks = true;
      fBatchCondition.notify_all();
   }

   // Wait untill all tasks are handled, then join the threads
   void DeActivate()
   {
      std::cout << "RBatchLoader => DeActivate" << std::endl;
      accept_tasks = false;
      fBatchCondition.notify_all();
   }

   /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   // Getters and Setters
   /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

   // Create a batch filled with the events on the given idx
   std::unique_ptr<TMVA::Experimental::RTensor<float>>
   CreateBatch(const TMVA::Experimental::RTensor<float> &chunkTensor, std::vector<size_t> idx)
   {
      auto batch = std::unique_ptr<TMVA::Experimental::RTensor<float>>(
         new TMVA::Experimental::RTensor<float>({fBatchSize, fNumColumns}));
      for (size_t i = 0; i < fBatchSize; i++) {
         std::copy(chunkTensor.GetData() + (idx[i] * fNumColumns), chunkTensor.GetData() + ((idx[i] + 1) * fNumColumns),
                   batch->GetData() + i * fNumColumns);
      }

      return batch;
   }

   // Add new tasks based on the given chunk_tensor
   void CreateTrainingBatches(const TMVA::Experimental::RTensor<float> &chunkTensor, std::vector<size_t> rowOrder,
                              bool shuffle = true)
   {
      // Wait until less than a full chunk of batches are in the queue before loading splitting the next chunk into
      // batches
      std::unique_lock<std::mutex> lock(fBatchLock);
      fBatchCondition.wait(lock, [this]() { return (fTrainingBatchQueue.size() < fMaxBatches) || !accept_tasks; });
      if (!accept_tasks) {
         return;
      }

      lock.unlock();

      if (shuffle) {
         std::shuffle(rowOrder.begin(), rowOrder.end(), fRng); // Shuffle the order of idx
      }

      std::vector<std::unique_ptr<TMVA::Experimental::RTensor<float>>> batches;

      // Create tasks of fBatchSize untill all idx are used
      for (size_t start = 0; (start + fBatchSize) <= rowOrder.size(); start += fBatchSize) {

         // Grab the first fBatchSize indices from the
         std::vector<size_t> idx;
         for (size_t i = start; i < (start + fBatchSize); i++) {
            idx.push_back(rowOrder[i]);
         }

         // Fill a batch
         batches.emplace_back(CreateBatch(chunkTensor, idx));
      }

      std::unique_lock<std::mutex> lock_2(fBatchLock);
      for (size_t i = 0; i < batches.size(); i++) {
         fTrainingBatchQueue.push(std::move(batches[i]));
      }

      lock_2.unlock();
      fBatchCondition.notify_one();
   }

   // Add new tasks based on the given chunk_tensor
   void CreateValidationBatches(const TMVA::Experimental::RTensor<float> &chunkTensor, std::vector<size_t> rowOrder)
   {

      // Create tasks of fBatchSize untill all idx are used
      for (size_t start = 0; (start + fBatchSize) <= rowOrder.size(); start += fBatchSize) {

         std::vector<size_t> idx;

         for (size_t i = start; i < (start + fBatchSize); i++) {
            idx.push_back(rowOrder[i]);
         }

         fValidationBatches.emplace_back(CreateBatch(chunkTensor, idx));
      }
   }

   // Set the validation index to 0 when starting the validation process
   void StartValidation() { fValidationIdx = 0; }
};

} // namespace Experimental
} // namespace TMVA

#endif // TMVA_RBatchLoader