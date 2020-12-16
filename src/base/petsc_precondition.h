#ifndef AETHER_BASE_PRECONDITION_H_
#define AETHER_BASE_PRECONDITION_H_

namespace aether::PETScWrappers {

/**
 * A very thin wrapper around deal.II's PETScWrappers::MPI::PreconditionNone 
 * which also works on any copy assignable vector type.
 */
class PreconditionNone : public dealii::PETScWrappers::PreconditionNone {
 public:
  using dealii::PETScWrappers::PreconditionNone::PreconditionNone;
  template <class VectorType>
  void vmult(VectorType &dst, const VectorType &src) const;
  void vmult(dealii::PETScWrappers::VectorBase &dst,
             const dealii::PETScWrappers::VectorBase &src) const;
};

template <class VectorType>
void PreconditionNone::vmult(VectorType &dst, const VectorType &src) const {
  dst = src;
}

inline void PreconditionNone::vmult(
    dealii::PETScWrappers::VectorBase &dst,
    const dealii::PETScWrappers::VectorBase &src) const {
  dealii::PETScWrappers::PreconditionNone::vmult(dst, src);
}

}  // namespace aether::PETScWrappers

#endif  // AETHER_BASE_PRECONDITION_H_