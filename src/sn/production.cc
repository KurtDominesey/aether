#include "production.h"

namespace aether::sn {

template <int dim>
Production<dim>::Production(const dealii::DoFHandler<dim> &dof_handler,
                            const std::vector<std::vector<double>> &nu_fission)
    : dof_handler(dof_handler), nu_fission(nu_fission) {}

template <int dim>
void Production<dim>::vmult(dealii::Vector<double> &dst, 
                            const dealii::BlockVector<double> &src) const {
  dst= 0;
  vmult_add(dst, src);
}

template <int dim>
void Production<dim>::vmult_add(dealii::Vector<double> &dst, 
                                const dealii::BlockVector<double> &src) const {
  const int num_groups = nu_fission.size();
  AssertDimension(src.n_blocks(), num_groups);
  const int dofs_per_cell = dof_handler.get_fe().dofs_per_cell;
  std::vector<dealii::types::global_dof_index> dof_indices(dofs_per_cell);
  for (const auto &cell : dof_handler.active_cell_iterators()) {
    cell->get_dof_indices(dof_indices);
    for (int g = 0; g < num_groups; ++g) {
      double nu_fission_gj = nu_fission[g][cell->material_id()];
      for (int i = 0; i < dofs_per_cell; ++i) {
        dst[dof_indices[i]] += nu_fission_gj * src.block(g)[dof_indices[i]];
      }
    }
  }
}

// These transposed vmults are essentially identical to the vmults of 
// Emission. Refactoring to avoid code duplication may be warranted.

template <int dim>
void Production<dim>::Tvmult(dealii::BlockVector<double> &dst,
                             const dealii::Vector<double> &src) const {
  dst = 0;
  Tvmult_add(dst, src);
}

template <int dim>
void Production<dim>::Tvmult_add(dealii::BlockVector<double> &dst, 
                                 const dealii::Vector<double> &src) const {
  const int num_groups = nu_fission.size();
  AssertDimension(dst.n_blocks(), num_groups);
  const int dofs_per_cell = dof_handler.get_fe().dofs_per_cell;
  std::vector<dealii::types::global_dof_index> dof_indices(dofs_per_cell);
  for (const auto &cell : dof_handler.active_cell_iterators()) {
    cell->get_dof_indices(dof_indices);
    for (int g = 0; g < num_groups; ++g) {
      double nu_fission_gj = nu_fission[g][cell->material_id()];
      for (int i = 0; i < dofs_per_cell; ++i) {
        dst.block(g)[dof_indices[i]] += nu_fission_gj * src[dof_indices[i]];
      }
    }
  }
}

template class Production<1>;
template class Production<2>;
template class Production<3>;

}  // namespace aether::sn