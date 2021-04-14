#include "base/petsc_solver.h"

namespace aether::PETScWrappers {

SolverFGMRES::AdditionalData::AdditionalData(
    const unsigned int restart_parameter)
    : restart_parameter(restart_parameter) {}

SolverFGMRES::SolverFGMRES(dealii::SolverControl &control,
                           const MPI_Comm &mpi_communicator,
                           const AdditionalData &data)
  : SolverBase(control, mpi_communicator), additional_data(data) {}

void SolverFGMRES::set_solver_type(KSP &ksp) const {
  PetscErrorCode ierr = KSPSetType(ksp, KSPFGMRES);
  AssertThrow(ierr == 0, dealii::ExcPETScError(ierr));
  // See dealii::SolverGMRES::set_solver_type for an explanation of this ugly
  // code. Workaround for KSPGMRESSetRestart.
  int (*func_ptr)(KSP, int);
  ierr = PetscObjectQueryFunction(reinterpret_cast<PetscObject>(ksp),
                                  "KSPGMRESSetRestart_C",
                                  reinterpret_cast<void (**)()>(&func_ptr));
  AssertThrow(ierr == 0, dealii::ExcPETScError(ierr));
  ierr = (*func_ptr)(ksp, additional_data.restart_parameter);
  AssertThrow(ierr == 0, dealii::ExcPETScError(ierr));
  // Only right preconditioning.
  ierr = KSPSetPCSide(ksp, PC_RIGHT);
  AssertThrow(ierr == 0, dealii::ExcPETScError(ierr));
  // Allow initial guess in solution vector.
  ierr = KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);
  AssertThrow(ierr == 0, dealii::ExcPETScError(ierr));
}

}  // namespace aether::PETScWrappers