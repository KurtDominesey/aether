#ifndef AETHER_PGD_SN_INNER_PRODUCTS_H_
#define AETHER_PGD_SN_INNER_PRODUCTS_H_

#include <valarray>

#include <deal.II/base/table.h>

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

template <int groupsND, int zonesND>
struct InnerProducts2D1D {
  using ArrayG = std::array<std::array<double, zonesND>, groupsND>;
  using ArrayGG = std::array<ArrayG, groupsND>;
  double leakage_co;  // coaxial or coplanar
  double leakage_trans;  // transverse
  ArrayG total;
  ArrayG fission;
  ArrayGG scattering;
};

}  // namespace aether::pgd::sn

#endif  // AETHER_PGD_SN_INNER_PRODUCTS_H_