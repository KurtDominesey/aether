#ifndef AETHER_BASE_PETSC_MATRIX_FREE_H_
#define AETHER_BASE_PETSC_MATRIX_FREE_H_

#include <deal.II/lac/petsc_matrix_free.h>

namespace aether::PETScWrappers {

/**
 * A thin wrapper around deal.II's PETScWrappers::MatrixFree which provides
 * a copy constructor and copy assignment operator. Also implements vmult_add,
 * Tvmult, and Tvmult_add, though each just throws an exception if called.
 */
class MatrixFree : public dealii::PETScWrappers::MatrixFree {
 public:
  using dealii::PETScWrappers::MatrixFree::MatrixFree;
  MatrixFree(const MatrixFree& other);
  MatrixFree& operator=(const MatrixFree &other);
  virtual void vmult_add(dealii::PETScWrappers::VectorBase &dst,
                         const dealii::PETScWrappers::VectorBase &src) const;
  virtual void Tvmult(dealii::PETScWrappers::VectorBase &dst,
                      const dealii::PETScWrappers::VectorBase &src) const;
  virtual void Tvmult_add(dealii::PETScWrappers::VectorBase &dst,
                          const dealii::PETScWrappers::VectorBase &src) const;
};

}  // namespace aether::PETScWrappers

#endif  // AETHER_BASE_PETSC_MATRIX_FREE_H_