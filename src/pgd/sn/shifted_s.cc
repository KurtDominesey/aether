#include "pgd/sn/shifted_s.h"

namespace aether::pgd::sn {

template <int dim, int qdim>
ShiftedS<dim, qdim>::ShiftedS(const FissionS<dim, qdim> &fission_s,
                              const FixedSourceS<dim, qdim> &fixed_source_s)
    : fission_s(fission_s), fixed_source_s(fixed_source_s) {}


template <int dim, int qdim>
void ShiftedS<dim, qdim>::vmult(dealii::BlockVector<double> &dst,
                                const dealii::BlockVector<double> &src) const {
  dealii::BlockVector<double> tmp(dst);
  fixed_source_s.vmult(dst, src);
  fission_s.vmult(tmp, src);
  dst.add(-shift, tmp);
}

template class ShiftedS<1>;
template class ShiftedS<2>;
template class ShiftedS<3>;

}  // namespace aether::pgd::sn