#include "sweeper.hpp"

void Sweeper::initialize(std::vector<int> &octant_to_global_,
                         dealii::BlockVector<double> &dst_ref, 
                         const dealii::BlockVector<double> &src_ref,
                         int dofs_per_cell) {
  octant_to_global = octant_to_global_;
  AssertDimension(dst_ref.n_blocks(), src_ref.n_blocks());
  dst = &dst_ref;
  src = &src_ref;
  // assume all cells have same number of dofs
  matrices.resize(octant_to_global.size(), 
                  dealii::FullMatrix<double>(dofs_per_cell));
  vectors.resize(octant_to_global.size(), 
                 dealii::Vector<double>(dofs_per_cell));
}