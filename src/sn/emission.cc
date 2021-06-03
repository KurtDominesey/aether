#include "emission.h"

namespace aether::sn {

template <int dim>
Emission<dim>::Emission(const dealii::DoFHandler<dim> &dof_handler,
                        const std::vector<std::vector<double>> &chi)
    : dof_handler(dof_handler), chi(chi) {}

template <int dim>
void Emission<dim>::vmult(dealii::BlockVector<double> &dst,
                          const dealii::Vector<double> &src) const {
  dst = 0;
  vmult_add(dst, src);
}

template <int dim>
void Emission<dim>::vmult_add(dealii::BlockVector<double> &dst, 
                              const dealii::Vector<double> &src) const {
  const int num_groups = chi.size();
  AssertDimension(dst.n_blocks(), num_groups);
  const int dofs_per_cell = dof_handler.get_fe().dofs_per_cell;
  std::vector<dealii::types::global_dof_index> dof_indices(dofs_per_cell);
  for (const auto &cell : dof_handler.active_cell_iterators()) {
    cell->get_dof_indices(dof_indices);
    for (int g = 0; g < num_groups; ++g) {
      double chi_gj = chi[g][cell->material_id()];
      for (int i = 0; i < dofs_per_cell; ++i) {
        dst.block(g)[dof_indices[i]] += chi_gj * src[dof_indices[i]];
      }
    }
  }
}

// These transposed vmults are essentially identical to the vmults of 
// Production. Refactoring to avoid code duplication may be warranted.

template <int dim>
void Emission<dim>::Tvmult(dealii::Vector<double> &dst, 
                           const dealii::BlockVector<double> &src) const {
  dst= 0;
  Tvmult_add(dst, src);
}

template <int dim>
void Emission<dim>::Tvmult_add(dealii::Vector<double> &dst, 
                                const dealii::BlockVector<double> &src) const {
  const int num_groups = chi.size();
  AssertDimension(src.n_blocks(), num_groups);
  const int dofs_per_cell = dof_handler.get_fe().dofs_per_cell;
  std::vector<dealii::types::global_dof_index> dof_indices(dofs_per_cell);
  for (const auto &cell : dof_handler.active_cell_iterators()) {
    cell->get_dof_indices(dof_indices);
    for (int g = 0; g < num_groups; ++g) {
      double chi_gj = chi[g][cell->material_id()];
      for (int i = 0; i < dofs_per_cell; ++i) {
        dst[dof_indices[i]] += chi_gj * src.block(g)[dof_indices[i]];
      }
    }
  }
}

template class Emission<1>;
template class Emission<2>;
template class Emission<3>;

}  // namespace aether::sn