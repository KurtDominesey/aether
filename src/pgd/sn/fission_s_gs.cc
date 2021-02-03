#include "pgd/sn/fission_s_gs.h"

namespace aether::pgd::sn {

template <int dim, int qdim>
void FissionSGS<dim, qdim>::vmult(
    dealii::BlockVector<double> &dst,
    const dealii::BlockVector<double> &src) const {
  AssertThrow(shifted, dealii::ExcMessage("Shift not set"))
  FixedSourceSGS<dim, qdim>::vmult(dst, src);
}

template <int dim, int qdim>
void FissionSGS<dim, qdim>::set_cross_sections(
    const std::vector<std::vector<Mgxs>> &mgxs) {
  FixedSourceSGS<dim, qdim>::set_cross_sections(mgxs);
  shifted = false;
}

template <int dim, int qdim>
void FissionSGS<dim, qdim>::set_shift(const double shift_) {
  AssertThrow(!shifted, dealii::ExcMessage("Already shifted"));
  shifted = true;
  shift = shift_;
  const int num_modes = this->mgxs_pseudos.size();
  const int num_groups = this->mgxs_pseudos[0][0].total.size();
  const int num_materials = this->mgxs_pseudos[0][0].total[0].size();
  for (int m = 0; m < num_modes; ++m) {
    for (int mp = 0; mp <= m; ++mp) {
      for (int g = 0; g < num_groups; ++g) {
        for (int gp = 0; gp < num_groups; ++gp) {
          if (m == mp && gp > g)
            break;
          for (int j = 0; j < num_materials; ++j) {
            this->mgxs_pseudos[m][mp].scatter[g][gp][j] += 
                this->mgxs_pseudos[m][mp].chi[g][j] *
                this->mgxs_pseudos[m][mp].nu_fission[gp][j] /
                shift;
          }
        }
      }
    }
  }
}

template class FissionSGS<1>;
template class FissionSGS<2>;
template class FissionSGS<3>;

}