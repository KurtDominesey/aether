#ifndef AETHER_PGD_SN_TRANSPORT_H_
#define AETHER_PGD_SN_TRANSPORT_H_

#include "sn/transport.h"

namespace aether::pgd::sn {

template <int dim, int qdim = dim == 1 ? 1 : 2>
class Transport : public aether::sn::Transport<dim, qdim> {
 public:
  using aether::sn::Transport<dim, qdim>::Transport;  // inherit constructors
  using Ordinate = typename aether::sn::Transport<dim, qdim>::Ordinate;
  void stream(dealii::Vector<double> &dst, 
              const dealii::Vector<double> &src,
              const std::vector<dealii::BlockVector<double>>
                  &boundary_conditions) const;
  void stream(dealii::BlockVector<double> &dst,
              const dealii::BlockVector<double> &src,
              const std::vector<dealii::BlockVector<double>>
                  &boundary_conditions) const;
  void collide(dealii::Vector<double> &dst,
               const dealii::Vector<double> &src) const;
  void collide(dealii::BlockVector<double> &dst,
               const dealii::BlockVector<double> &src) const;
};

}  // namespace aether::pgd::sn

#endif  // AETHER_PGD_SN_TRANSPORT_H_