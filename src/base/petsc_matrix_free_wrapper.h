#ifndef AETHER_BASE_PETSC_MATRIX_FREE_WRAPPER_H_
#define AETHER_BASE_PETSC_MATRIX_FREE_WRAPPER_H_

#include "base/petsc_matrix_free.h"

namespace aether::PETScWrappers {

template <class MatrixType>
class MatrixFreeWrapper : public MatrixFree {
 public:
  MatrixFreeWrapper(const MPI_Comm& communicator, 
                    const unsigned int m, const unsigned int n, 
                    const unsigned int local_rows, 
                    const unsigned int local_columns,
                    const MatrixType& matrix);
  template <class VectorType>
  void vmult(VectorType &dst, const VectorType &src) const;
  void vmult(dealii::PETScWrappers::VectorBase &dst,
             const dealii::PETScWrappers::VectorBase &src) const = 0;
 protected:
  const MatrixType& matrix;
};

template <class MatrixType>
MatrixFreeWrapper<MatrixType>::MatrixFreeWrapper(
    const MPI_Comm& communicator, const unsigned int m, const unsigned int n,
    const unsigned int local_rows, const unsigned int local_columns, 
    const MatrixType& matrix)
    : MatrixFree(communicator, m, n, local_rows, local_columns), matrix(matrix)
  {}

template <class MatrixType>
template <class VectorType>
void MatrixFreeWrapper<MatrixType>::vmult(VectorType &dst, 
                                          const VectorType &src) const {
  matrix.vmult(dst, src);
}

}  // namesapce aether::PETScWrappers

#endif  // AETHER_BASE_PETSC_MATRIX_FREE_WRAPPER_H_