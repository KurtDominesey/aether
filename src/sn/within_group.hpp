#ifndef AETHER_SN_WITHIN_GROUP_H_
#define AETHER_SN_WITHIN_GROUP_H_

#include <deal.II/lac/block_linear_operator.h>

#include "transport.hpp"
#include "scattering.hpp"
#include "moment_to_discrete.hpp"
#include "discrete_to_moment.hpp"

template <int dim, int qdim = dim == 1 ? 1 : 2>
class WithinGroup {
 public:
  WithinGroup(Transport<dim, qdim> &transport,
              MomentToDiscrete<qdim> &m2d,
              Scattering<dim> &scattering,
              DiscreteToMoment<qdim> &d2m);
  void vmult(dealii::BlockVector<double> &dst,
             const dealii::BlockVector<double> &src) const;
  void Tvmult(dealii::BlockVector<double> &dst,
              const dealii::BlockVector<double> &src) const;
 protected:
  Transport<dim, qdim> &transport;
  MomentToDiscrete<qdim> &m2d;
  Scattering<dim> &scattering;
  DiscreteToMoment<qdim> &d2m;

 private:
  // dealii::BlockVector<double> flux_m;
  // dealii::BlockVector<double> scattered;
  // dealii::BlockVector<double> src_total;
};

// template <//typename Range = dealii::BlockVector<double>, typename Domain = Range,
//           // typename BlockPayload = dealii::internal::BlockLinearOperatorImplementation::EmptyBlockPayload<>,
//           int dim, int qdim>
// void within_group(
//     Transport<dim, qdim> transport, MomentToDiscrete<qdim> moment_to_discrete,
//     Scattering<dim> scattering, DiscreteToMoment<qdim> discrete_to_moment) {
//   // auto Linv = dealii::linear_operator(transport);
//   // auto M2D = dealii::linear_operator(moment_to_discrete);
//   dealii::LinearOperator<dealii::BlockVector<double>> S 
//       = dealii::linear_operator(scattering);
//   // auto D2M = dealii::linear_operator(discrete_to_moment);
//   // auto wg = Linv * D2M * S * M2D;
// }

#endif  // AETHER_SN_WITHIN_GROUP_H_