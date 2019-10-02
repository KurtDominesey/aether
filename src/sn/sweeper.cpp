#include "sweeper.hpp"

void Sweeper::initialize(std::vector<int> &octant_to_global,
                         dealii::BlockVector<double> &dst_ref, 
                         dealii::BlockVector<double> &src_ref) {
  octant_to_global = octant_to_global;
  AssertDimension(dst_ref.n_blocks(), src_ref.n_blocks());
  dst = &dst_ref;
  src = &src_ref;    
}

template <class DOFINFO>
void Sweeper::initialize_info(DOFINFO &info, bool face) const {
  info.initialize_matrices(octant_to_global.size(), face);
  info.initialize_vectors(octant_to_global.size(), face);
}

template <class DOFINFO>
void Sweeper::assemble(const DOFINFO &info) {
  for (int n = 0; n < info.n_matrices(); ++n) {
    dealii::FullMatrix<double> &local_matrix = info.matrix(n).matrix;
    dealii::Vector<double> &local_src = info.vector(n).block(0);
    dealii::Vector<double> local_dst = local_src;
    local_matrix.gauss_jordan();  // directly invert
    local_matrix.vmult(local_dst, local_src);
    dst->block(n).add(info.indices, local_dst);
  }
}

template <class DOFINFO>
void Sweeper::assemble(const DOFINFO &info1, const DOFINFO &info2) {}
// do not do any global work solely on face contributions