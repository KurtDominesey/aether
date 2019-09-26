#ifndef AETHER_SN_SCATTERING_H_
#define AETHER_SN_SCATTERING_H_

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/lac/block_vector.h>

template <int dim>
class Scattering {
 public:
  /**
   * Constructor.
   * 
   * @param Scattering material cross-sections.
   */
  Scattering(dealii::DoFHandler<dim> &dof_handler,
             std::vector<double> &cross_sections);
  /**
   * Apply the linear operator.
   * 
   * @param dst Destination vector.
   * @param src Source vector.
   */
  void vmult(dealii::BlockVector<double> &dst,
             const dealii::BlockVector<double> &src) const;
  /**
   * Apply the transpose of the linear operator (not implemented)
   * 
   * @param dst Destination vector.
   * @param src Source vector.
   */
  void Tvmult(dealii::BlockVector<double> &dst,
              const dealii::BlockVector<double> &src) const;

 protected:
  std::vector<double> &cross_sections;
  dealii::DoFHandler<dim> &dof_handler;
};

template <int dim>
Scattering<dim>::Scattering(dealii::DoFHandler<dim> &dof_handler,
                            std::vector<double> &cross_sections)
    : dof_handler(dof_handler), cross_sections(cross_sections) {}

template <int dim>
void Scattering<dim>::vmult(dealii::BlockVector<double> &dst,
                  const dealii::BlockVector<double> &src) const {
  dst = 0;
  const int num_ell = 1;
  const int dofs_per_cell = dof_handler.get_fe().dofs_per_cell;
  std::vector<dealii::types::global_dof_index> dof_indices(dofs_per_cell);
  for (const auto &cell : dof_handler.cell_iterators()) {
    int material = cell->material_id();
    cell->get_dof_indices(dof_indices);
    for (int ell = 0, lm = 0; ell < num_ell; ++ell) {
      double cross_section = cross_sections[material*num_ell+ell];
      for (int m = -ell; m <= ell; ++m, ++lm) {
        for (int i = 0; i < dofs_per_cell; ++i) {
          dst.block(lm)[dof_indices[i]] =
              cross_section * src.block(lm)[dof_indices[i]];
        }
      }
    }
  }
}

#endif  // AETHER_SN_SCATTERING_H_