#ifndef AETHER_SN_MOMENT_TO_DISCRETE_H_
#define AETHER_SN_MOMENT_TO_DISCRETE_H_

#include <deal.II/base/quadrature.h>
#include <deal.II/lac/block_vector.h>

#include "sn/quadrature.h"

namespace aether::sn {

/**
 * Moment-to-discrete operator.
 * 
 * The Legendre order is not fixed, but is rather determined by the size or
 * number of blocks of the destination vector `dst` passed to
 * @ref MomentToDiscrete::vmult.
 */
template <int dim, int qdim = dim == 1 ? 1 : 2>
class MomentToDiscrete {
 public:
  /**
   * Contructor.
   */
  MomentToDiscrete(const QAngle<dim, qdim> &quadrature);
  /**
   * Matrix-vector multiplication.
   */
  void vmult(dealii::Vector<double> &dst, 
             const dealii::Vector<double> &src) const;
  /**
   * Matrix-vector multiplication.
   */
  void vmult(dealii::BlockVector<double> &dst, 
             const dealii::BlockVector<double> &src) const;
  /**
   * Adding matrix-vector mutliplcation.
   */
  void vmult_add(dealii::Vector<double> &dst, 
                 const dealii::Vector<double> &src) const;
  /**
   * Adding matrix-vector multiplication.
   */
  void vmult_add(dealii::BlockVector<double> &dst, 
                 const dealii::BlockVector<double> &src) const;

 protected:
  //! Angular quadrature.
  const QAngle<dim, qdim> &quadrature;
};

}  // namespace aether::sn

#endif  // AETHER_SN_MOMENT_TO_DISCRETE_H_