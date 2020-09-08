#ifndef AETHER_SN_DISCRETE_TO_MOMENT_H_
#define AETHER_SN_DISCRETE_TO_MOMENT_H_

#include <deal.II/base/quadrature.h>
#include <deal.II/lac/block_vector.h>

#include "sn/quadrature.h"
#include "sn/spherical_harmonics.h"

namespace aether::sn {

template <int dim, int qdim = dim == 1 ? 1 : 2>
class DiscreteToMoment {
 public:
  DiscreteToMoment(const QAngle<dim, qdim> &quadrature);
  void vmult(dealii::Vector<double> &dst,
             const dealii::Vector<double> &src) const;
  void vmult(dealii::BlockVector<double> &dst, 
             const dealii::BlockVector<double> &src) const;
  void vmult_add(dealii::Vector<double> &dst,
                 const dealii::Vector<double> &src) const;
  void vmult_add(dealii::BlockVector<double> &dst, 
                 const dealii::BlockVector<double> &src) const;
  int n_block_rows(int order) const;
  int n_block_cols() const;
  void discrete_to_legendre(dealii::Vector<double> &dst,
                            const dealii::Vector<double> &src) const;
  void discrete_to_legendre(dealii::BlockVector<double> &dst,
                            const dealii::BlockVector<double> &src) const;
  void moment_to_legendre(dealii::Vector<double> &dst,
                          const dealii::Vector<double> &moments,
                          const int order) const;
  void moment_to_legendre(dealii::BlockVector<double> &dst,
                          const dealii::BlockVector<double> &moments) const;

 protected:
  const QAngle<dim, qdim> &quadrature;
};

int num_moments(int order, int dim);

int legendre_order(int num_moments, int dim);

int num_moments_of_order(int ell, int dim);

}  // namespace aether::sn

#endif  // AETHER_SN_DISCRETE_TO_MOMENT_H_
