#include "petsc_block_block_vector.h"

namespace aether::PETScWrappers::MPI {

void BlockBlockVector::reinit(const unsigned int n_blocks) {
  std::vector<size_type> block_sizes(n_blocks, 0);
  this->block_indices.reinit(block_sizes);
  if (this->components.size() != this->n_blocks()) {
    this->components.resize(this->n_blocks());
    collect_sizes();
  }
  for (size_type i = 0; i < this->n_blocks(); ++i) {
    this->components[i].reinit(0, MPI_COMM_SELF, 0, 0);
    collect_sizes();
  }
}

}  // namespace aether::PETScWrappers::MPI