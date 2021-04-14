#ifndef AETHER_BASE_PETSC_SOLVER_H_
#define AETHER_BASE_PETSC_SOLVER_H_

#include <deal.II/lac/petsc_solver.h>

namespace aether::PETScWrappers {

class SolverFGMRES : public dealii::PETScWrappers::SolverBase {
 public:
  struct AdditionalData {
    AdditionalData(const unsigned int restart_parameter=30);
    unsigned int restart_parameter;
  };
  SolverFGMRES(dealii::SolverControl &control,
               const MPI_Comm &mpi_communicator=PETSC_COMM_SELF,
               const AdditionalData &data=AdditionalData());

 protected:
  const AdditionalData additional_data;
  virtual void set_solver_type(KSP &ksp) const override;
};

}  // namespace aether::PETScWrappers

#endif  // AETHER_BASE_PETSC_SOLVER_H_