#ifndef AETHER_SN_SCATTERING_H_
#define AETHER_SN_SCATTERING_H_

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_vector.h>

template <int dim>
class Scattering {
 public:
  /**
   * Constructor.
   * 
   * @param Scattering material cross-sections.
   */
  Scattering(dealii::DoFHandler<dim> &dof_handler);
  /**
   * Apply the linear operator.
   * 
   * @param dst Destination vector.
   * @param src Source vector.
   */
  void vmult(dealii::Vector<double> &dst,
             const dealii::Vector<double> &src,
             const std::vector<double> &cross_sections) const;
  /**
   * Apply the linear operator.
   * 
   * @param dst Destination vector.
   * @param src Source vector.
   */
  void vmult(dealii::BlockVector<double> &dst,
             const dealii::BlockVector<double> &src,
             const std::vector<double> &cross_sections) const;
  /**
   * Apply the transpose of the linear operator (not implemented)
   * 
   * @param dst Destination vector.
   * @param src Source vector.
   */
  void Tvmult(dealii::BlockVector<double> &dst,
              const dealii::BlockVector<double> &src) const;
  /**
   * Add the linear operator.
   * 
   * @param dst Destination vector.
   * @param src Source vector.
   */
  void vmult_add(dealii::Vector<double> &dst,
                 const dealii::Vector<double> &src,
                 const std::vector<double> &cross_sections) const;
  /**
   * Add the linear operator.
   * 
   * @param dst Destination vector.
   * @param src Source vector.
   */
  void vmult_add(dealii::BlockVector<double> &dst,
                 const dealii::BlockVector<double> &src,
                 const std::vector<double> &cross_sections) const;

 protected:
  dealii::DoFHandler<dim> &dof_handler;
};

#endif  // AETHER_SN_SCATTERING_H_