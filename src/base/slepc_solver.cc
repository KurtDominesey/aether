#include "base/slepc_solver.h"

namespace aether::SLEPcWrappers {

SolverRayleigh::SolverRayleigh(dealii::SolverControl &control,
                               const MPI_Comm &mpi_comm,
                               const AdditionalData &data)
    : dealii::SLEPcWrappers::SolverPower(control, mpi_comm, data) {
  PetscErrorCode ierr = EPSPowerSetShiftType(eps, EPS_POWER_SHIFT_RAYLEIGH);
  AssertThrow(ierr == 0, dealii::SLEPcWrappers::SolverBase::ExcSLEPcError(ierr));
}

}  // namespace aether::SLEPcWrappers