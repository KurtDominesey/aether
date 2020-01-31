#ifndef AETHER_SN_QUADRATURE_LIB_H_
#define AETHER_SN_QUADRATURE_LIB_H_

#include "sn/quadrature.h"

namespace aether::sn {

template <int dim, int qdim = dim == 1 ? 1 : 2>
class QPglc : public QAngle<dim, qdim> {
 public:
  QPglc(const int num_polar, const int num_azim = 0);
};

}  // namespace aether::sn

#endif  // AETHER_SN_QUADRATURE_LIB_H_