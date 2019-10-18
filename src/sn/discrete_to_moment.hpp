#ifndef AETHER_SN_DISCRETE_TO_MOMENT_H_
#define AETHER_SN_DISCRETE_TO_MOMENT_H_

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/lac/block_vector.h>

template <int qdim>
class DiscreteToMoment {
 public:
  DiscreteToMoment(const dealii::Quadrature<qdim> &quadrature);
  void vmult(dealii::BlockVector<double> &dst, 
             const dealii::BlockVector<double> &src) const;
  void Tvmult(dealii::BlockVector<double> &dst,
              const dealii::BlockVector<double> &src) const;
  void vmult_add(dealii::BlockVector<double> &dst, 
                 const dealii::BlockVector<double> &src) const;
  void Tvmult_add(dealii::BlockVector<double> &dst,
                  const dealii::BlockVector<double> &src) const;

 protected:
  const dealii::Quadrature<qdim> &quadrature;
};

#endif  // AETHER_SN_DISCRETE_TO_MOMENT_H_
