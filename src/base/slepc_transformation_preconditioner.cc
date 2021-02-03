#include "base/slepc_transformation_preconditioner.h"

namespace aether::SLEPcWrappers {

TransformationPreconditioner::TransformationPreconditioner(
    const MPI_Comm &mpi_communicator,
    const dealii::PETScWrappers::MatrixBase &matrix_) 
    : dealii::SLEPcWrappers::TransformationBase(mpi_communicator) {
  matrix = static_cast<Mat>(matrix_);
  PetscErrorCode ierr = STSetType(st, const_cast<char*>(STPRECOND));
  using ExcSLEPcError = dealii::SLEPcWrappers::SolverBase::ExcSLEPcError;
  AssertThrow(ierr == 0, ExcSLEPcError(ierr));
  ierr = STPrecondSetMatForPC(st, matrix);
  AssertThrow(ierr == 0, ExcSLEPcError(ierr));
}


}  // namespace aether::SLEPcWrappers