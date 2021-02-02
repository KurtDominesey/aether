#include "base/petsc_precondition_matrix.h"

namespace aether::PETScWrappers {

PreconditionerMatrix::PreconditionerMatrix(
    const dealii::PETScWrappers::MatrixBase &matrix) {
  initialize(matrix);
}

void PreconditionerMatrix::initialize(
    const dealii::PETScWrappers::MatrixBase &matrix_) {
  clear();
  matrix = static_cast<Mat>(matrix_);
  create_pc();
  AssertThrow(pc != nullptr, dealii::StandardExceptions::ExcInvalidState());
  PetscErrorCode ierr = PCSetType(pc, const_cast<char*>(PCMAT));
  AssertThrow(ierr == 0, dealii::ExcPETScError(ierr));
  ierr = PCSetFromOptions(pc);
  AssertThrow(ierr == 0, dealii::ExcPETScError(ierr));
  ierr = PCSetUp(pc);
  AssertThrow(ierr == 0, dealii::ExcPETScError(ierr));
}

}