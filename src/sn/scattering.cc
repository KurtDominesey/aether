#include "scattering.h"

namespace aether::sn {

template <int dim>
Scattering<dim>::Scattering(const dealii::DoFHandler<dim> &dof_handler)
    : dof_handler(dof_handler) {}

template <int dim>
void Scattering<dim>::vmult(dealii::Vector<double> &dst,
                            const dealii::Vector<double> &src,
                            const std::vector<double> &cross_sections) const {
  dealii::BlockVector<double> dst_b(1, dof_handler.n_dofs());
  dealii::BlockVector<double> src_b(1, dof_handler.n_dofs());
  src_b = src;
  vmult(dst_b, src_b, cross_sections);
  dst = dst_b;
}

template <int dim>
void Scattering<dim>::vmult_add(
    dealii::Vector<double> &dst, 
    const dealii::Vector<double> &src,
    const std::vector<double> &cross_sections) const {
  dealii::BlockVector<double> dst_b(1, dof_handler.n_dofs());
  dealii::BlockVector<double> src_b(1, dof_handler.n_dofs());
  dst_b = dst;
  src_b = src;
  vmult_add(dst_b, src_b, cross_sections);
  dst = dst_b;
}

template <int dim>
template <class Vector>
void Scattering<dim>::vmult(
    dealii::BlockVectorBase<Vector> &dst,
    const dealii::BlockVectorBase<Vector> &src,
    const std::vector<double> &cross_sections) const {
  dst = 0;
  vmult_add(dst, src, cross_sections);
}

template <int dim>
template <class Vector>
void Scattering<dim>::vmult_add(
    dealii::BlockVectorBase<Vector> &dst,
    const dealii::BlockVectorBase<Vector> &src,
    const std::vector<double> &cross_sections) const {
  const int num_ell = 1;
  const int dofs_per_cell = dof_handler.get_fe().dofs_per_cell;
  std::vector<dealii::types::global_dof_index> dof_indices(dofs_per_cell);
  for (const auto &cell : dof_handler.active_cell_iterators()) {
    int material = cell->material_id();
    cell->get_dof_indices(dof_indices);
    for (int ell = 0, lm = 0; ell < num_ell; ++ell) {
      double cross_section = cross_sections[material*num_ell+ell];
      for (int m = -ell; m <= ell; ++m, ++lm) {
        for (int i = 0; i < dofs_per_cell; ++i) {
          dst.block(lm)[dof_indices[i]] +=
              cross_section * src.block(lm)[dof_indices[i]];
        }
      }
    }
  }
}

template class Scattering<1>;
template class Scattering<2>;
template class Scattering<3>;

// Vector is dealii::Vector<double>
template void Scattering<1>::vmult<dealii::Vector<double>>(
    dealii::BlockVectorBase<dealii::Vector<double>>&,
    const dealii::BlockVectorBase<dealii::Vector<double>>&,
    const std::vector<double>&) const;
template void Scattering<2>::vmult<dealii::Vector<double>>(
    dealii::BlockVectorBase<dealii::Vector<double>>&,
    const dealii::BlockVectorBase<dealii::Vector<double>>&,
    const std::vector<double>&) const;
template void Scattering<3>::vmult<dealii::Vector<double>>(
    dealii::BlockVectorBase<dealii::Vector<double>>&,
    const dealii::BlockVectorBase<dealii::Vector<double>>&,
    const std::vector<double>&) const;
// Vector is dealii::PETScWrappers::MPI::Vector
template void Scattering<1>::vmult<dealii::PETScWrappers::MPI::Vector>(
    dealii::BlockVectorBase<dealii::PETScWrappers::MPI::Vector>&,
    const dealii::BlockVectorBase<dealii::PETScWrappers::MPI::Vector>&,
    const std::vector<double>&) const;
template void Scattering<2>::vmult<dealii::PETScWrappers::MPI::Vector>(
    dealii::BlockVectorBase<dealii::PETScWrappers::MPI::Vector>&,
    const dealii::BlockVectorBase<dealii::PETScWrappers::MPI::Vector>&,
    const std::vector<double>&) const;
template void Scattering<3>::vmult<dealii::PETScWrappers::MPI::Vector>(
    dealii::BlockVectorBase<dealii::PETScWrappers::MPI::Vector>&,
    const dealii::BlockVectorBase<dealii::PETScWrappers::MPI::Vector>&,
    const std::vector<double>&) const;

}  // namespace aether::sn