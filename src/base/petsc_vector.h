#ifndef AETHER_BASE_PETSC_VECTOR_H_
#define AETHER_BASE_PETSC_VECTOR_H_

#include <deal.II/lac/petsc_vector.h>

namespace aether::PETScWrappers::MPI {

/**
 * A very thin wrapper around deal.II's PETScWrappers::MPI::Vector with
 * constructors that match those of deal.II's native serial vectors.
 */
class Vector : public dealii::PETScWrappers::MPI::Vector {
 public:
  using BaseClass = dealii::PETScWrappers::MPI::Vector;
  using size_type = BaseClass::size_type;
  using BaseClass::Vector;
  Vector(const size_type size);
  Vector& operator=(const BaseClass::value_type value);
  Vector& operator=(const Vector &other);
  using BaseClass::reinit;
  void reinit(const size_type size, const bool omit_zeroing_entries = false);
};

inline Vector::Vector(const size_type size) 
    : BaseClass(MPI_COMM_WORLD, size, size) {}

inline void Vector::reinit(const size_type size, 
                           const bool omit_zeroing_entries) {
  BaseClass::reinit(MPI_COMM_WORLD, size, size, omit_zeroing_entries);
}

inline Vector& Vector::operator=(const BaseClass::value_type value) {
  BaseClass::operator=(value);
  return *this;
}

inline Vector& Vector::operator=(const Vector &other) {
  BaseClass::operator=(other);
  return *this;
}

}  // namespace aether::PETScWrappers::MPI

#endif  // AETHER_BASE_PETSC_VECTOR_H_