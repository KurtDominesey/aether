#ifndef AETHER_PGD_SN_INNER_PRODUCTS_H_
#define AETHER_PGD_SN_INNER_PRODUCTS_H_

#include <valarray>

#include <deal.II/base/table.h>

namespace aether::pgd::sn {

struct InnerProducts {
  InnerProducts(int num_materials, int num_legendre)
      : collision(num_materials),
        scattering(std::valarray<double>(num_legendre), num_materials) {};
  double streaming;
  std::valarray<double> collision;
  std::valarray<std::valarray<double>> scattering;
  InnerProducts& operator=(const double &value) {
    streaming = value;
    collision = value;
    for (int material = 0; material < scattering.size(); ++material)
      scattering[material] = value;
    return *this;
  }
  InnerProducts& operator*=(const double &value) {
    streaming *= value;
    collision *= value;
    for (int material = 0; material < scattering.size(); ++material)
      scattering[material] *= value;
    return *this;
  }
  InnerProducts& operator*=(const InnerProducts &other) {
    streaming *= other.streaming;
    collision *= other.collision;
    scattering *= other.scattering;
    return *this;
  }
};

}  // namespace aether::pgd::sn

#endif  // AETHER_PGD_SN_INNER_PRODUCTS_H_