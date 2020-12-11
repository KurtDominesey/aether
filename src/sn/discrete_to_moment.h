#ifndef AETHER_SN_DISCRETE_TO_MOMENT_H_
#define AETHER_SN_DISCRETE_TO_MOMENT_H_

#include <deal.II/base/quadrature.h>
#include <deal.II/lac/block_vector_base.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/petsc_vector.h>

#include "sn/quadrature.h"
#include "sn/spherical_harmonics.h"

namespace aether::sn {

/**
 * Discrete-to-moment operator.
 * 
 * The Legendre order is not fixed, but is rather determined by the size or
 * number of blocks of the destination vector `dst` passed to
 * @ref DiscreteToMoment::vmult.
 */
template <int dim, int qdim = dim == 1 ? 1 : 2>
class DiscreteToMoment {
 public:
  /**
   * Constructor.
   */
  DiscreteToMoment(const QAngle<dim, qdim> &quadrature);
  /**
   * Matrix-vector multiplcation.
   */
  void vmult(dealii::Vector<double> &dst,
             const dealii::Vector<double> &src) const;
  /**
   * Matrix-vector multiplication.
   */
  template <class Vector>
  void vmult(dealii::BlockVectorBase<Vector> &dst, 
             const dealii::BlockVectorBase<Vector> &src) const;
  /**
   * Adding matrix-vector multiplication.
   */
  void vmult_add(dealii::Vector<double> &dst,
                 const dealii::Vector<double> &src) const;
  /**
   * Adding matrix-vector multiplication.
   */
  template <class Vector>
  void vmult_add(dealii::BlockVectorBase<Vector> &dst, 
                 const dealii::BlockVectorBase<Vector> &src) const;
  /**
   * Number of block rows (moments) for a given Legendre `order`.
   * 
   * Dimension `dim` | Order \f$\ell\f$
   * :-------------: | -----------------
   * 1D              | \f$\ell+1\f$
   * 2D              | \f$(\ell+1)(\ell+2)/2\f$
   * 3D              | \f$(\ell+1)^2\f$
   */
  int n_block_rows(int order) const;
  /**
   * Number of block columns (ordinates).
   * 
   * Equal to the size of @ref DiscreteToMoment::quadrature.
   */
  int n_block_cols() const;
  /**
   * Discrete-to-Legendre conversion.
   * 
   * The source vector is converted to moments then passed to 
   * @ref DiscreteToMoment::moment_to_legendre.
   */
  void discrete_to_legendre(dealii::Vector<double> &dst,
                            const dealii::Vector<double> &src) const;
  /**
   * Discrete-to-Legendre conversion.
   */
  void discrete_to_legendre(dealii::BlockVector<double> &dst,
                            const dealii::BlockVector<double> &src) const;
  /**
   * Discrete-to-Legendre conversion.
   * 
   * The Legendre flux is defined as
   * \f$\verb|dst|_\ell=\sqrt{\sum_{m=-\ell}^{\ell}\verb|src|^2_{\ell,m}}\f$,
   * with the zero moments omitted (see @ref DiscreteToMoment::n_block_rows).
   */
  void moment_to_legendre(dealii::Vector<double> &dst,
                          const dealii::Vector<double> &moments,
                          const int order) const;
  /**
   * Discrete-to-Legendre conversion.
   */
  void moment_to_legendre(dealii::BlockVector<double> &dst,
                          const dealii::BlockVector<double> &moments) const;

 protected:
  //! Angular quadrature.
  const QAngle<dim, qdim> &quadrature;
};

}  // namespace aether::sn

#endif  // AETHER_SN_DISCRETE_TO_MOMENT_H_
