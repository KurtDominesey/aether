#ifndef AETHER_SN_QUADRATURE_LIB_H_
#define AETHER_SN_QUADRATURE_LIB_H_

#include "sn/quadrature.h"

namespace aether::sn {

template <int dim, int qdim = dim == 1 ? 1 : 2>
class QPglc : public QAngle<dim, qdim> {
 public:
  QPglc() = default;
  QPglc(const int num_polar, const int num_azim = 0);
  int reflected_index(
      const int n, const dealii::Tensor<1, dim> &normal) const override;
};

}  // namespace aether::sn

#endif  // AETHER_SN_QUADRATURE_LIB_H_