#ifndef AETHER_PGD_SN_FISSION_S_H_
#define AETHER_PGD_SN_FISSION_S_H_

#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_vector.h>

#include "sn/moment_to_discrete.h"
#include "sn/emission.h"
#include "sn/production.h"
#include "sn/discrete_to_moment.h"
#include "pgd/sn/transport.h"

namespace aether::pgd::sn {

template <int dim, int qdim = dim == 1 ? 1 : 2>
class FissionS {
 public:
  FissionS(const Transport<dim, qdim> &transport,
           const aether::sn::MomentToDiscrete<dim, qdim> &m2d,
           const std::vector<std::vector<aether::sn::Emission<dim>>> &emission,
           const std::vector<std::vector<aether::sn::Production<dim>>> &production,
           const aether::sn::DiscreteToMoment<dim, qdim> &d2m);
  void vmult(dealii::BlockVector<double> &dst,
             const dealii::BlockVector<double> &src,
             bool transposing=false) const;
  bool transposed = false;
  
 protected:
  const Transport<dim, qdim> &transport;
  const aether::sn::MomentToDiscrete<dim, qdim> &m2d;
  const std::vector<std::vector<aether::sn::Emission<dim>>> &emission;
  const std::vector<std::vector<aether::sn::Production<dim>>> &production;
  const aether::sn::DiscreteToMoment<dim, qdim> &d2m;
};

}

#endif  // AETHER_PGD_SN_FISSION_S_H_