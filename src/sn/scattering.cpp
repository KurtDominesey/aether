#include "scattering.hpp"

template <int dim>
Scattering<dim>::Scattering(dealii::DoFHandler<dim> &dof_handler,
                            std::vector<double> &cross_sections)
    : dof_handler(dof_handler), cross_sections(cross_sections) {}

template <int dim>
void Scattering<dim>::vmult(dealii::Vector<double> &dst,
                            const dealii::Vector<double> &src) const {
  dealii::BlockVector<double> dst_b(1, dof_handler.n_dofs());
  dealii::BlockVector<double> src_b(1, dof_handler.n_dofs());
  src_b = src;
  vmult(dst_b, src_b);
  dst = dst_b;
}

template <int dim>
void Scattering<dim>::vmult_add(dealii::Vector<double> &dst,
                                const dealii::Vector<double> &src) const {
  dealii::BlockVector<double> dst_b(1, dof_handler.n_dofs());
  dealii::BlockVector<double> src_b(1, dof_handler.n_dofs());
  dst_b = dst;
  src_b = src;
  vmult_add(dst_b, src_b);
  dst = dst_b;
}

template <int dim>
void Scattering<dim>::vmult(dealii::BlockVector<double> &dst,
                  const dealii::BlockVector<double> &src) const {
  dst = 0;
  vmult_add(dst, src);
}

template <int dim>
void Scattering<dim>::vmult_add(dealii::BlockVector<double> &dst,
                  const dealii::BlockVector<double> &src) const {
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