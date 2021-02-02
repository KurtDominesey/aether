#ifndef AETHER_BASE_PRECONDITION_SHELL_H_
#define AETHER_BASE_PRECONDITION_SHELL_H_

#include <deal.II/lac/petsc_vector_base.h>
#include <deal.II/lac/petsc_matrix_base.h>
#include <deal.II/lac/petsc_precondition.h>

namespace aether::PETScWrappers {

class PreconditionerShell : public dealii::PETScWrappers::PreconditionerBase {
 public:
  PreconditionerShell() = default;
  PreconditionerShell(const dealii::PETScWrappers::MatrixBase &matrix);
  void initialize(const dealii::PETScWrappers::MatrixBase &matrix);

 protected:
  static PetscErrorCode apply(PC pc, Vec x, Vec y);
};

}  // namespace aether::PETScWrappers

#endif  // AETHER_BASE_PRECONDITION_SHELL_H_