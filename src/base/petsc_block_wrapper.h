#ifndef AETHER_BASE_PETSC_BLOCK_WRAPPER_H_
#define AETHER_BASE_PETSC_BLOCK_WRAPPER_H_

#include <deal.II/lac/petsc_vector_base.h>
#include <deal.II/lac/block_vector.h>

#include "base/petsc_matrix_free_wrapper.h"

namespace aether::PETScWrappers {

template <class MatrixType>
class BlockWrapper : public MatrixFreeWrapperBase<MatrixType> {
 public:
  BlockWrapper(const int num_blocks, const MPI_Comm& communicator, 
               const int block_size, const int local_size,
               const MatrixType& matrix);
  using MatrixFreeWrapperBase<MatrixType>::vmult;
  void vmult(dealii::PETScWrappers::VectorBase &dst,
             const dealii::PETScWrappers::VectorBase &src) const;

 protected:
  const int num_blocks;
  const MPI_Comm& communicator;
  const int block_size;
  const int local_size;
};

template <class MatrixType>
BlockWrapper<MatrixType>::BlockWrapper(const int num_blocks, 
                                       const MPI_Comm& communicator,
                                       const int block_size,
                                       const int local_size,
                                       const MatrixType& matrix)
    : MatrixFreeWrapperBase<MatrixType>(communicator,
                                        num_blocks*block_size,
                                        num_blocks*block_size,
                                        num_blocks*local_size,
                                        num_blocks*local_size,
                                        matrix),
      num_blocks(num_blocks), communicator(communicator), 
      block_size(block_size), local_size(local_size) {}


template <class MatrixType>
void BlockWrapper<MatrixType>::vmult(
    dealii::PETScWrappers::VectorBase &dst,
    const dealii::PETScWrappers::VectorBase &src) const {
  dealii::BlockVector<double> dst_b(num_blocks, block_size);
  dealii::BlockVector<double> src_b(num_blocks, block_size);
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

#endif  // AETHER_BASE_PETSC_BLOCK_WRAPPER_H_