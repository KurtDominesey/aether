#ifndef AETHER_BASE_PETSC_WRAPPER_H_
#define AETHER_BASE_PETSC_WRAPPER_H_

#include <deal.II/lac/petsc_vector_base.h>
#include <deal.II/lac/block_vector.h>

#include "base/petsc_matrix_free_wrapper.h"

namespace aether::PETScWrappers {

template <class MatrixType>
class MatrixFreeWrapper : public MatrixFreeWrapperBase<MatrixType> {
 public:
  using MatrixFreeWrapperBase<MatrixType>::MatrixFreeWrapperBase;
  using MatrixFreeWrapperBase<MatrixType>::vmult;
  void vmult(dealii::PETScWrappers::VectorBase &dst,
             const dealii::PETScWrappers::VectorBase &src) const override;
};



template <class MatrixType>
void MatrixFreeWrapper<MatrixType>::vmult(
    dealii::PETScWrappers::VectorBase &dst, 
    const dealii::PETScWrappers::VectorBase &src) const {
  const int size = dst.size();
  AssertDimension(size, src.size());
  dealii::Vector<double> dst_d(size);
  dealii::Vector<double> src_d(size);
  for (int i = 0; i < size; ++i) {
    dst_d[i] = dst[i];
    src_d[i] = src[i];
  }
  vmult(dst_d, src_d);
  for (int i = 0; i < size; ++i)
    dst[i] = dst_d[i];
}

}  // namespace aether::PETScWrappers

#endif  // AETHER_BASE_PETSC_WRAPPER_H_