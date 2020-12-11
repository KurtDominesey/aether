#include "base/petsc_matrix_free.h"

namespace aether::PETScWrappers {

MatrixFree::MatrixFree(const MatrixFree& other) {
  operator=(other);
}

MatrixFree& MatrixFree::operator=(const MatrixFree &other) {
  reinit(other.get_mpi_communicator(), other.m(), other.n(), 
         other.local_size(), other.local_size());
}

void MatrixFree::vmult_add(
    dealii::PETScWrappers::VectorBase &dst,
    const dealii::PETScWrappers::VectorBase &src) const {
  AssertThrow(false, dealii::ExcNotImplemented());
}

void MatrixFree::Tvmult(
    dealii::PETScWrappers::VectorBase &dst,
    const dealii::PETScWrappers::VectorBase &src) const {
  AssertThrow(false, dealii::ExcNotImplemented());
}

void MatrixFree::Tvmult_add(
    dealii::PETScWrappers::VectorBase &dst,
    const dealii::PETScWrappers::VectorBase &src) const {
  AssertThrow(false, dealii::ExcNotImplemented());
}

}  // namespace aether::PETScWrappers