#ifndef AETHER_BASE_PRECONDITION_MATRIX_H_
#define AETHER_BASE_PRECONDITION_MATRIX_H_

#include <deal.II/lac/petsc_matrix_base.h>
#include <deal.II/lac/petsc_precondition.h>

namespace aether::PETScWrappers {

class PreconditionerMatrix : public dealii::PETScWrappers::PreconditionerBase {
 public:
  PreconditionerMatrix() = default;
  PreconditionerMatrix(const dealii::PETScWrappers::MatrixBase &matrix);
  void initialize(const dealii::PETScWrappers::MatrixBase &matrix);
};

}  // namespace aether::PETScWrappers

#endif  // AETHER_BASE_PRECONDITION_MATRIX_H_