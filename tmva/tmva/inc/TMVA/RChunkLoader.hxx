#ifndef TMVA_CHUNKLOADER
#define TMVA_CHUNKLOADER

#include <iostream>
#include <vector>

#include "TMVA/RTensor.hxx"
#include "ROOT/RDataFrame.hxx"
#include "ROOT/RVec.hxx"

namespace TMVA {
namespace Experimental {

// RChunkLoader class used to load content of a RDataFrame onto a RTensor.
template <typename First, typename... Rest>
class RChunkLoader {

private:
   size_t fOffset = 0, fVecSizeIdx = 0;
   std::vector<size_t> fMaxVecSizes;

   float fVecPadding;

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
   RChunkLoader(TMVA::Experimental::RTensor<float> &chunkTensor,
                std::vector<size_t> maxVecSizes = std::vector<size_t>(), float vecPadding = 0.0)
      : fChunkTensor(chunkTensor), fMaxVecSizes(maxVecSizes), fVecPadding(vecPadding)
   {
   }

   void operator()(First first, Rest... rest)
   {
      fVecSizeIdx = 0;
      AssignToTensor(std::forward<First>(first), std::forward<Rest>(rest)...);
   }
};

} // namespace Experimental
} // namespace TMVA

#endif // TMVA_CHUNKLOADER