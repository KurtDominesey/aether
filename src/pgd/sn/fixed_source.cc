#include "pgd/sn/fixed_source.hpp"

namespace aether::pgd::sn {

template <int dim, int qdim>
void FixedSource<dim, qdim>::update_cache(
      const dealii::BlockVector<double> &mode) {
  cache.modes.push_back(mode);
  const dealii::BlockIndices block_indices(mode.get_block_indices());
  cache.streamed.emplace_back(block_indices);
  cache.collided.emplace_back(block_indices);
  cache.scattered.emplace_back(block_indices);
  stream(cache.streamed.back(), mode);
  collide(cache.collided.back(), mode);
  scatter(cache.scattered.back(), mode);
}

template <int dim, int qdim>
void FixedSource<dim, qdim>::stream(dealii::BlockVector<double> &dst,
                                    const dealii::BlockVector<double> &src) 
                                    const {
  AssertDimension(dst.n_blocks(), src.n_blocks());
  Assert(dst.n_blocks() == 1, dealii::ExcNotImplemented());
  for (int g = 0; g < src.n_blocks(); ++g) {
    auto transport = dynamic_cast<Transport<dim>&>(
        this->within_groups[g].transport.transport);
    transport.stream(dst.block(g), src.block(g));
  }
}

template <int dim, int qdim>
void FixedSource<dim, qdim>::collide(dealii::BlockVector<double> &dst,
                                     const dealii::BlockVector<double> &src) 
                                     const {
  AssertDimension(dst.n_blocks(), src.n_blocks());
  Assert(dst.n_blocks() == 1, dealii::ExcNotImplemented());
  for (int g = 0; g < src.n_blocks(); ++g) {
    auto transport = dynamic_cast<Transport<dim>&>(
        this->within_groups[g].transport.transport);
    transport.collide(dst.block(g), src.block(g));
  }
}

template <int dim, int qdim>
void FixedSource<dim, qdim>::scatter(dealii::BlockVector<double> &dst,
                                     const dealii::BlockVector<double> &src)
                                     const {
  AssertDimension(dst.n_blocks(), src.n_blocks());
  dealii::BlockVector<double> dst_m(1, src.block(0).size());
  for (int g = 0; g < src.n_blocks(); ++g) {
    this->d2m.vmult(dst_m, src.block(g));
  }
}

}  // namespace aehter::pgd::sn
