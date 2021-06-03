#ifndef AETHER_SN_EMISSION_H_
#define AETHER_SN_EMISSION_H_

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_vector.h>

namespace aether::sn {

/**
 * Emission of neutrons produced by fission.
 */
template <int dim>
class Emission {
 public:
  Emission(const dealii::DoFHandler<dim> &dof_handler,
           const std::vector<std::vector<double>> &chi);
  void vmult(dealii::BlockVector<double> &dst,
             const dealii::Vector<double> &src) const;
  void vmult_add(dealii::BlockVector<double> &dst,
                 const dealii::Vector<double> &src) const;
  void Tvmult(dealii::Vector<double> &dst, 
              const dealii::BlockVector<double> &src) const;
  void Tvmult_add(dealii::Vector<double> &dst, 
                  const dealii::BlockVector<double> &src) const;

 protected:
  const dealii::DoFHandler<dim> &dof_handler;
  const std::vector<std::vector<double>> &chi;
};

}  // namespace aether::sn

#endif  // AETHER_SN_EMISSION_H_