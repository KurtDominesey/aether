#ifndef AETHER_BASE_SLEPC_TRANSFORMATION_PRECONDITIONER_H_
#define AETHER_BASE_SLEPC_TRANSFORMATION_PRECONDITIONER_H_

#include <deal.II/lac/slepc_spectral_transformation.h>
#include <deal.II/lac/slepc_solver.h>
#include <deal.II/lac/petsc_matrix_base.h>

namespace aether::SLEPcWrappers {

class TransformationPreconditioner : 
    public dealii::SLEPcWrappers::TransformationBase {
 public:
  TransformationPreconditioner(
      const MPI_Comm &mpi_communicator,
      const dealii::PETScWrappers::MatrixBase &matrix);

 protected:
  Mat matrix;
};


}  // namespace aether::SLEPcWrappers

#endif  // AETHER_BASE_SLEPC_TRANSFORMATION_PRECONDITIONER_H_