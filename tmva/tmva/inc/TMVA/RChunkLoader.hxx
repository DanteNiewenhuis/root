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
   size_t offset = 0, fVecSizeIdx = 0;
   std::vector<size_t> fVecSizes;

   TMVA::Experimental::RTensor<float> &fchunkTensor;

   /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   // Value assigning
   /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

   // Load the final given value into fchunkTensor
   // Add a label to the end of the row if given
   template <typename First_T>
   void AssignToTensor(First_T first)
   {
      fchunkTensor.GetData()[offset++] = first;
   }
   // Vector version of the previous function
   template <typename VecType>
   void AssignToTensor(ROOT::RVec<VecType> first)
   {
      AssignVector(first);
   }

   // Recursively loop through the given values, and load them onto the fchunkTensor
   template <typename First_T, typename... Rest_T>
   void AssignToTensor(First_T first, Rest_T... rest)
   {
      fchunkTensor.GetData()[offset++] = first;

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
      size_t vec_size = fVecSizes[fVecSizeIdx++];

      for (size_t i = 0; i < vec_size; i++) {
         fchunkTensor.GetData()[offset++] = vec[i];
      }
   }

public:
   /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   // Constructor
   /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   RChunkLoader(TMVA::Experimental::RTensor<float> &chunkTensor, std::vector<size_t> vecSizes = std::vector<size_t>())
      : fchunkTensor(chunkTensor), fVecSizes(vecSizes)
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