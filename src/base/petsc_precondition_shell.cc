#include "base/petsc_precondition_shell.h"

namespace aether::PETScWrappers {

PreconditionerShell::PreconditionerShell(
    const dealii::PETScWrappers::MatrixBase &matrix) {
  initialize(matrix);
}

void PreconditionerShell::initialize(
    const dealii::PETScWrappers::MatrixBase &matrix_) {
  clear();
  matrix = static_cast<Mat>(matrix_);
  create_pc();
  AssertThrow(pc != nullptr, dealii::StandardExceptions::ExcInvalidState());
  PetscErrorCode ierr = PCSetType(pc, const_cast<char*>(PCSHELL));
  AssertThrow(ierr == 0, dealii::ExcPETScError(ierr));
  ierr = PCShellSetContext(pc, matrix);
  AssertThrow(ierr == 0, dealii::ExcPETScError(ierr));
  ierr = PCShellSetApply(pc, PreconditionerShell::apply);
  AssertThrow(ierr == 0, dealii::ExcPETScError(ierr));
}

PetscErrorCode PreconditionerShell::apply(PC pc, Vec x, Vec y) {
  Mat matrix;
  PetscErrorCode ierr = PCShellGetContext(pc, (void**)&matrix);
  AssertThrow(ierr == 0, dealii::ExcPETScError(ierr));
  MatMult(matrix, x, y);
  return 0;
}

}