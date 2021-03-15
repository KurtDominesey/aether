#ifndef AETHER_PGD_SN_FIXED_SOURCE_S_H_
#define AETHER_PGD_SN_FIXED_SOURCE_S_H_

#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_vector.h>

#include "sn/fixed_source.h"
#include "pgd/sn/transport_block.h"
#include "pgd/sn/inner_products.h"

namespace aether::pgd::sn {

template <int dim, int qdim = dim == 1 ? 1 : 2>
class FixedSourceS {
 public:
  FixedSourceS(
      const std::vector<std::vector<aether::sn::FixedSource<dim, qdim>>> &blocks,
      const aether::sn::MomentToDiscrete<dim, qdim> &m2d,
      const aether::sn::DiscreteToMoment<dim, qdim> &d2m,
      const Mgxs &mgxs);
  void vmult(dealii::Vector<double> &dst,
             const dealii::Vector<double> &src) const;
  void vmult(dealii::BlockVector<double> &dst,
             const dealii::BlockVector<double> &src) const;
  void get_inner_products_lhs(
      std::vector<std::vector<InnerProducts>> &inner_products,
      const dealii::BlockVector<double> &modes);
  void get_inner_products_rhs(std::vector<double>& inner_products,
                              const dealii::BlockVector<double> &modes);
  std::vector<std::vector<double>> streaming;  // streaming coefficients

 protected:
  // friend class FixedSourceSProblem<dim, qdim>;
  const std::vector<std::vector<aether::sn::FixedSource<dim, qdim>>> &blocks;
  //! Moment to discrete operator, \f$M\f$
  const aether::sn::MomentToDiscrete<dim, qdim> &m2d;
  //! Discrete to moment operator, \f$D\f$
  const aether::sn::DiscreteToMoment<dim, qdim> &d2m;
  //! Multigroup cross-sections for computing inner products
  const Mgxs &mgxs;
};

}

#endif  // AETHER_PGD_SN_FIXED_SOURCE_S_H_