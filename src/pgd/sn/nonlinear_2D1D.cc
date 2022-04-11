#include "pgd/sn/nonlinear_2D1D.h"

namespace aether::pgd::sn {

Nonlinear2D1D::Nonlinear2D1D(
    FixedSource2D1D<1> &one_d, FixedSource2D1D<2> &two_d, 
    const std::vector<std::vector<int>> &materials, const Mgxs &mgxs, 
    bool both_mg) 
    : one_d(one_d), two_d(two_d), materials(materials), mgxs(mgxs),
      both_mg(both_mg) {}

void Nonlinear2D1D::enrich() {
  one_d.enrich();
  two_d.enrich();
  one_d.normalize(both_mg);
  two_d.normalize(both_mg);
  one_d.set_inner_prods();
  two_d.set_inner_prods();
}

double Nonlinear2D1D::iter() {
  // one_d.normalize(both_mg);
  one_d.set_inner_prods();
  two_d.setup(one_d.iprods_flux, one_d.iprods_src, materials, mgxs);
  double r2 = two_d.solve();
  two_d.normalize(both_mg);
  two_d.set_inner_prods();
  one_d.setup(two_d.iprods_flux, two_d.iprods_src, materials, mgxs);
  double r1 = one_d.solve();
  double r = std::sqrt(r2*r2+r1*r1);
  std::cout << "2D: " << r2 << "  "
            << "1D: " << r1 << "  "
            << "2D/1D: " << r << "\n";
  return r;
}

void Nonlinear2D1D::reweight() {
  AssertThrow(false, dealii::ExcNotImplemented());
  return;
  const int num_modes = one_d.prods.size();
  const int len = both_mg ? mgxs.num_groups : 1;
  matrix_wgt.grow_or_shrink(matrix_wgt.m()+len);
  weights.grow_or_shrink(weights.size()+len);
  src_wgt.grow_or_shrink(src_wgt.size()+len);
  for (int m = 0; m < num_modes; ++m)
    one_d.normalize(both_mg, false, m);
  // one_d.normalize(both_mg);
  one_d.set_inner_prods();
  const int num_groups1 = one_d.iprods_flux[0].rxn.num_groups;
  const int num_groups2 = two_d.iprods_flux[0].rxn.num_groups;
  const int last = (num_modes - 1) * len;
  // Fill in last elements of source
  for (int s = 0; s < one_d.iprods_src.size(); ++s) {
    for (int g = 0; g < mgxs.num_groups; ++g) {
      int g1 = num_groups1 > 1 ? g : 0;
      int g2 = num_groups2 > 1 ? g : 0;
      int gg = both_mg ? g : 0;
      src_wgt[last+gg] += one_d.iprods_src[s][g1] * two_d.iprods_src[s][g2];
    }
  }
  InnerProducts2D1D iprod1(num_groups1, materials.size());
  InnerProducts2D1D iprod2(num_groups2, materials[0].size());
  // Fill in last elements of matrix
  for (int m = 0; m < num_modes; ++m) {
    int mm = m * len;
    if (m < num_modes - 1) {
      one_d.set_inner_prod_flux(
          one_d.prods[m].test, one_d.prods.back(), iprod1);
      two_d.set_inner_prod_flux(
          two_d.prods[m].test, two_d.prods.back(), iprod2);
    }
    for (int g = 0; g < mgxs.num_groups; ++g) {
      // Fill in last row
      int g1 = num_groups1 > 1 ? g : 0;
      int g2 = num_groups2 > 1 ? g : 0;
      int gg = both_mg ? g : 0;
      matrix_wgt(last+gg, mm+gg) +=
          one_d.iprods_flux[m].stream_co[g1] *
          two_d.iprods_flux[m].stream_trans[g2] +
          one_d.iprods_flux[m].stream_trans[g1] *
          two_d.iprods_flux[m].stream_co[g2];
      for (int i = 0; i < materials.size(); ++i) {
        for (int j = 0; j < materials[i].size(); ++j) {
          int matl = materials[i][j];
          matrix_wgt(last+gg, mm+gg) +=
              mgxs.total[g][matl] *
              one_d.iprods_flux[m].rxn.total[g1][i] *
              two_d.iprods_flux[m].rxn.total[g2][j];
          for (int gp = 0; gp < mgxs.num_groups; ++gp) {
            int gp1 = num_groups1 > 1 ? gp : 0;
            int gp2 = num_groups2 > 1 ? gp : 0;
            int ggp = both_mg ? gp : 0;
            matrix_wgt(last+gg, mm+ggp) -=
                mgxs.scatter[g][gp][matl] *
                one_d.iprods_flux[m].rxn.scatter[g1][gp1][i] *
                two_d.iprods_flux[m].rxn.scatter[g2][gp2][j];
          }
        }
      }
      if (m == num_modes - 1)
        continue;
      // Fill in last column
      matrix_wgt(mm+gg, last+gg) +=
          iprod1.stream_co[g1] * 
          iprod2.stream_trans[g2] +
          iprod1.stream_trans[g1] * 
          iprod2.stream_co[g2];
      for (int i = 0; i < materials.size(); ++i) {
        for (int j = 0; j < materials[i].size(); ++j) {
          int matl = materials[i][j];
          matrix_wgt(mm+gg, last+gg) +=
              mgxs.total[g][matl] *
              iprod1.rxn.total[g1][i] *
              iprod2.rxn.total[g2][j];
          for (int gp = 0; gp < mgxs.num_groups; ++gp) {
            int gp1 = num_groups1 > 1 ? gp : 0;
            int gp2 = num_groups2 > 1 ? gp : 0;
            int ggp = both_mg ? gp : 0;
            matrix_wgt(last+gg, mm+ggp) -=
                mgxs.scatter[g][gp][matl] *
                iprod1.rxn.scatter[g1][gp1][i] *
                iprod2.rxn.scatter[g2][gp2][j];
          }
        }
      }
    }
  }
  // Solve for the new weights
  dealii::LAPACKFullMatrix<double> lu_wgt(matrix_wgt);
  lu_wgt.compute_lu_factorization();
  weights = src_wgt;
  lu_wgt.solve(src_wgt);
  dealii::swap(weights, src_wgt);
  std::cout << "weights: " << weights << "\n";
  // should get rid of this needless renormalization and rescaling
  for (int m = 0; m < num_modes; ++m) {
    if (m < num_modes - 1)
      one_d.normalize(both_mg, false, m);
    if (both_mg) {
      for (int g = 0; g < mgxs.num_groups; ++g) {
        // one_d.prods[m].scale(weights[m*len+g], g);
        one_d.prods[m].psi *= weights[m*len+g];
        one_d.prods[m].phi *= weights[m*len+g];
        one_d.prods[m].streamed *= weights[m*len+g];
      }
    } else {
      // one_d.prods[m].scale(weights[m]);
      one_d.prods[m].psi *= weights[m];
      one_d.prods[m].phi *= weights[m];
      one_d.prods[m].streamed *= weights[m];  
    }
  }
}

}