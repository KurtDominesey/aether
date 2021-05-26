#include "pgd/sn/fission_source_shifted_s.h"

namespace aether::pgd::sn {

template <int dim, int qdim>
FissionSourceShiftedS<dim, qdim>::FissionSourceShiftedS(
    const FissionS<dim, qdim> &fission_s,
    const FixedSourceS<dim, qdim> &fixed_source_s,
    const FixedSourceSGS<dim, qdim> &fixed_source_s_gs)
    : fission_s(fission_s), fixed_source_s(fixed_source_s), 
      fixed_source_s_gs(fixed_source_s_gs) {}


template <int dim, int qdim>
void FissionSourceShiftedS<dim, qdim>::vmult(
    dealii::BlockVector<double> &dst,
    const dealii::BlockVector<double> &src) const {
  dst = 0;
  dealii::BlockVector<double> ax(dst);
  fission_s.vmult(ax, src);
  dealii::IterationNumberControl control(10, 0);
  control.enable_history_data();
  dealii::SolverFGMRES<dealii::BlockVector<double>> solver(control);
  solver.solve(fixed_source_s, dst, ax, fixed_source_s_gs);
  dst.add(-shift, src);
}

template class FissionSourceShiftedS<1>;
template class FissionSourceShiftedS<2>;
template class FissionSourceShiftedS<3>;

}  // namespace aether::pgd::sn