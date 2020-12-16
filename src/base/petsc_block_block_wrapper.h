#ifndef AETHER_BASE_PETSC_BLOCK_BLOCK_WRAPPER_H_
#define AETHER_BASE_PETSC_BLOCK_BLOCK_WRAPPER_H_

#include <deal.II/lac/block_vector.h>

#include "base/petsc_matrix_free_wrapper.h"

namespace aether::PETScWrappers {

template <class MatrixType>
class BlockBlockWrapper : public MatrixFreeWrapper<MatrixType> {
 public:
  BlockBlockWrapper(const int num_blocks, 
                    const int num_subblocks, 
                    const MPI_Comm& communicator, 
                    const int block_size, 
                    const int local_size,
                    const MatrixType& matrix);
  using MatrixFreeWrapper<MatrixType>::vmult;
  void vmult(dealii::PETScWrappers::VectorBase &dst,
             const dealii::PETScWrappers::VectorBase &src) const;
 protected:
  const int num_blocks;
  const int num_subblocks;
  const MPI_Comm& communicator;
  const int block_size;
  const int local_size;
};

template <class MatrixType>
BlockBlockWrapper<MatrixType>::BlockBlockWrapper(const int num_blocks,
                                                 const int num_subblocks,
                                                 const MPI_Comm& communicator,
                                                 const int block_size,
                                                 const int local_size,
                                                 const MatrixType& matrix)
  : MatrixFreeWrapper<MatrixType>(communicator, 
                                  num_blocks*num_subblocks*block_size, 
                                  num_blocks*num_subblocks*block_size,
                                  num_blocks*num_subblocks*local_size,
                                  num_blocks*num_subblocks*local_size,
                                  matrix),
    num_blocks(num_blocks), num_subblocks(num_subblocks), 
    communicator(communicator), block_size(block_size), local_size(local_size)
    {}

template <class MatrixType>
void BlockBlockWrapper<MatrixType>::vmult(
    dealii::PETScWrappers::VectorBase &dst, 
    const dealii::PETScWrappers::VectorBase &src) const {
  dealii::BlockVector<double> dst_b(num_blocks, num_subblocks*block_size);
  dealii::BlockVector<double> src_b(num_blocks, num_subblocks*block_size);
  const int size = dst.size();
  AssertDimension(size, src.size());
  AssertDimension(size, dst_b.size());
  for (int i = 0; i < size; ++i) {
    dst_b[i] = dst[i];
    src_b[i] = src[i];
  }
  vmult(dst_b, src_b);
  for (int i = 0; i < size; ++i)
    dst[i] = dst_b[i];
}

}  // namespace aether::PETScWrappers

#endif  // AETHER_BASE_PETSC_BLOCK_BLOCK_MATRIX_H_