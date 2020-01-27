#ifndef AETHER_PGD_SN_TRANSPORT_BLOCK_H_
#define AETHER_PGD_SN_TRANSPORT_BLOCK_H_

#include "pgd/sn/transport.h"
#include "sn/transport_block.h"

namespace aether::pgd::sn {

template <int dim, int qdim = dim == 1 ? 1 : 2>
class TransportBlock : public aether::sn::TransportBlock<dim, qdim> {
 public:
  using aether::sn::TransportBlock<dim, qdim>::TransportBlock;
  template <typename VectorType>
  void stream(VectorType &dst, const VectorType &src) const;
  template <typename VectorType>
  void collide(VectorType &dst, const VectorType &src) const;
};

template <int dim, int qdim>
template <typename VectorType>
void TransportBlock<dim, qdim>::stream(VectorType &dst,
                                       const VectorType &src) const {
  const Transport<dim, qdim> &transport =
      dynamic_cast<const Transport<dim, qdim>&>(this->transport);
  transport.stream(dst, src, this->boundary_conditions);
}

template <int dim, int qdim>
template <typename VectorType>
void TransportBlock<dim, qdim>::collide(VectorType &dst,
                                        const VectorType &src) const {
  const Transport<dim, qdim> &transport =
      dynamic_cast<const Transport<dim, qdim>&>(this->transport);
  transport.collide(dst, src);
}

template class TransportBlock<1>;
template class TransportBlock<2>;
template class TransportBlock<3>;

}  // namespace aether::pgd::sn

#endif  // AETHER_PGD_SN_TRANSPORT_BLOCK_H_