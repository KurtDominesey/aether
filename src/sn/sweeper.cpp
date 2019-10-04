#include "sweeper.hpp"

void Sweeper::initialize(std::vector<int> &octant_to_global,
                         dealii::BlockVector<double> &dst_ref, 
                         dealii::BlockVector<double> &src_ref) {
  octant_to_global = octant_to_global;
  AssertDimension(dst_ref.n_blocks(), src_ref.n_blocks());
  dst = &dst_ref;
  src = &src_ref;    
}