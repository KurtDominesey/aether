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

#endif  // AETHER_SN_SCATTERING_H_