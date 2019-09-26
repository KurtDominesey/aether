#ifndef AETHER_SN_MOMENT_TO_DISCRETE_H_
#define AETHER_SN_MOMENT_TO_DISCRETE_H_

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/lac/block_vector.h>

template <int qdim>
class MomentToDiscrete {
 public:
  MomentToDiscrete(const dealii::Quadrature<qdim> &quadrature);
  void vmult(dealii::BlockVector<double> &dst, 
             const dealii::BlockVector<double> &src) const;
  void Tvmult(dealii::BlockVector<double> &dst,
              const dealii::BlockVector<double> &src) const;

 protected:
  const dealii::Quadrature<qdim> &quadrature;
};

#endif  // AETHER_SN_MOMENT_TO_DISCRETE_H_