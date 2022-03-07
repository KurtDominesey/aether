#ifndef AETHER_BASE_SLEPC_SOLVER_H_
#define AETHER_BASE_SLEPC_SOLVER_H_

#include <deal.II/lac/slepc_solver.h>

#include <slepceps.h>

namespace aether::SLEPcWrappers {

class SolverRayleigh : public dealii::SLEPcWrappers::SolverPower {
 public:
  SolverRayleigh(dealii::SolverControl &control,
                 const MPI_Comm &mpi_comm=PETSC_COMM_SELF,
                 const AdditionalData &data=AdditionalData());

 private:
  using Base = dealii::SLEPcWrappers::SolverPower;
};

}  // namespace aether::SLEPcWrappers

#endif  // AETHER_BASE_SLEPC_SOLVER_H_