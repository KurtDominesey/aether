#ifndef AETHER_BLOCK_BLOCK_VECTOR_TEMPLATES_H_
#define AETHER_BLOCK_BLOCK_VECTOR_TEMPLATES_H_

#include "base/block_block_vector.h"

namespace aether {

template <typename Number>
BlockBlockVector<Number>::BlockBlockVector(const unsigned int n_blocks,
                                           const unsigned int n_subblocks,
                                           const size_type block_size) {
  reinit(n_blocks, n_subblocks, block_size);
}

template <typename Number>
BlockBlockVector<Number>::BlockBlockVector(
    const std::vector<std::vector<size_type>> &block_sizes) {
  reinit(block_sizes, false);
}

template <typename Number>
BlockBlockVector<Number>::BlockBlockVector(
    const BlockBlockVector<Number> &other) : BaseClass() {
  this->components.resize(other.n_blocks());
  this->block_indices = other.block_indices;
  for (size_type i = 0; i < this->n_blocks(); ++i)
    this->components[i] = other.components[i];
}

template <typename Number>
void BlockBlockVector<Number>::reinit(const unsigned int n_blocks,
                                      const unsigned int n_subblocks,
                                      const size_type block_size,
                                      const bool omit_zeroing_entries) {
  std::vector<std::vector<size_type>> block_sizes(n_blocks,
      std::vector<size_type>(n_subblocks, block_size));
  reinit(block_sizes, omit_zeroing_entries);
}

template <typename Number>
void BlockBlockVector<Number>::reinit(
    const std::vector<std::vector<size_type>> &block_sizes, 
    const bool omit_zeroing_entries) {
  std::vector<size_type> flat_sizes(block_sizes.size());
  for (size_type i = 0; i < block_sizes.size(); ++i)
    for (size_type j = 0; j < block_sizes[i].size(); ++j)
      flat_sizes[i] += block_sizes[i][j];
  this->block_indices.reinit(flat_sizes);
  if (this->components.size() != this->n_blocks())
    this->components.resize(this->n_blocks());
  for (size_type i = 0; i < this->n_blocks(); ++i) {
    this->components[i].reinit(block_sizes[i], omit_zeroing_entries);
  }
}

template <typename Number>
void BlockBlockVector<Number>::reinit(const BlockBlockVector<Number> &other,
                                      const bool omit_zeroing_entries) {
  this->block_indices = other.get_block_indices();
  if (this->components.size() != this->n_blocks())
    this->components.resize(this->n_blocks());
  for (size_type i = 0; i < this->n_blocks(); ++i)
    this->block(i).reinit(other.block(i), omit_zeroing_entries);
}

}  // namespace aether

#endif  // AETHER_BLOCK_BLOCK_VECTOR_TEMPLATES