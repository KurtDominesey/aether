#ifndef AETHER_SN_MOMENT_TO_DISCRETE_H_
#define AETHER_SN_MOMENT_TO_DISCRETE_H_

#include <deal.II/base/quadrature.h>
#include <deal.II/lac/block_vector.h>

#include "sn/quadrature.h"

namespace aether::sn {

template <int dim, int qdim = dim == 1 ? 1 : 2>
class MomentToDiscrete {
 public:
  MomentToDiscrete(const QAngle<dim, qdim> &quadrature);
  void vmult(dealii::Vector<double> &dst, 
             const dealii::Vector<double> &src) const;
  void vmult(dealii::BlockVector<double> &dst, 
             const dealii::BlockVector<double> &src) const;
  void vmult_add(dealii::Vector<double> &dst, 
                 const dealii::Vector<double> &src) const;
  void vmult_add(dealii::BlockVector<double> &dst, 
                 const dealii::BlockVector<double> &src) const;

 protected:
  const QAngle<dim, qdim> &quadrature;
};

}  // namespace aether::sn

#endif  // AETHER_SN_MOMENT_TO_DISCRETE_H_