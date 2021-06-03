#ifndef AETHER_SN_PRODUCTION_H_
#define AETHER_SN_PRODUCTION_H_

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_vector.h>

namespace aether::sn {

/**
 * Production of neutrons by fission.
 */
template <int dim>
class Production {
 public:
  Production(const dealii::DoFHandler<dim> &dof_handler,
             const std::vector<std::vector<double>> &nu_fission);
  void vmult(dealii::Vector<double> &dst, 
             const dealii::BlockVector<double> &src) const;
  void vmult_add(dealii::Vector<double> &dst, 
                 const dealii::BlockVector<double> &src) const;
  void Tvmult(dealii::BlockVector<double> &dst,
              const dealii::Vector<double> &src) const;
  void Tvmult_add(dealii::BlockVector<double> &dst,
                  const dealii::Vector<double> &src) const;

 protected:
  const dealii::DoFHandler<dim> &dof_handler;
  const std::vector<std::vector<double>> &nu_fission;
};

}  // namespace aether::sn

#endif  // AETHER_SN_PRODUCTION_H_