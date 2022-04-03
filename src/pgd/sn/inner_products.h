#ifndef AETHER_PGD_SN_INNER_PRODUCTS_H_
#define AETHER_PGD_SN_INNER_PRODUCTS_H_

#include <valarray>

#include <deal.II/base/table.h>

#include "base/mgxs.h"

namespace aether::pgd::sn {

struct InnerProducts {
  InnerProducts(int num_materials, int num_legendre)
      : collision(num_materials),
        fission(num_materials),
        scattering(std::valarray<double>(num_legendre), num_materials) {};
  double streaming;
  std::valarray<double> collision;
  std::valarray<double> fission;
  std::valarray<std::valarray<double>> scattering;
  InnerProducts& operator=(const double &value) {
    streaming = value;
    collision = value;
    fission = value;
    for (int material = 0; material < scattering.size(); ++material)
      scattering[material] = value;
    return *this;
  }
  InnerProducts& operator*=(const double &value) {
    streaming *= value;
    collision *= value;
    fission *= value;
    for (int material = 0; material < scattering.size(); ++material)
      scattering[material] *= value;
    return *this;
  }
  InnerProducts& operator*=(const InnerProducts &other) {
    streaming *= other.streaming;
    collision *= other.collision;
    scattering *= other.scattering;
    fission *= other.fission;
    return *this;
  }
  double eval() const {
    return streaming + collision.sum() + scattering.sum().sum();
  }
};

InnerProducts operator*(const InnerProducts &a, const InnerProducts&b);

struct InnerProducts2D1D {
  InnerProducts2D1D(int num_groups, int num_zones) :
      stream_co(num_groups), 
      stream_trans(num_groups), 
      rxn(num_groups, num_zones, 1) {};
  std::vector<double> stream_co;  // coaxial or coplanar
  std::vector<double> stream_trans;  // transverse
  Mgxs rxn;
  InnerProducts2D1D& operator=(const double v) {
    for (int g = 0; g < rxn.num_groups; ++g) {
      stream_co[g] = v;
      stream_trans[g] = v;
      for (int j = 0; j < rxn.num_materials; ++j) {
        rxn.total[g][j] = v;
        rxn.nu_fission[g][j] = v;
        rxn.chi[g][j] = v;
        for (int gp = 0; gp < rxn.num_groups; ++gp) {
          rxn.scatter[g][gp][j] = v;
        }
      }
    }
    return *this;
  }
};

}  // namespace aether::pgd::sn

#endif  // AETHER_PGD_SN_INNER_PRODUCTS_H_