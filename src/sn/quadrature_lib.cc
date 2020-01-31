#include "quadrature_lib.h"

namespace aether::sn {

template <>
QPglc<1, 1>::QPglc(const int num_polar, const int num_azim)
    : QAngle<1, 1>(dealii::QGauss<1>(2 * num_polar)) {
  Assert(num_azim == 0, dealii::ExcImpossibleInDim(1));
}

template <int dim, int qdim>
QPglc<dim, qdim>::QPglc(const int num_polar, const int num_azim)
    : QAngle<dim, qdim>(
          dealii::QGauss<1>(2 * num_polar),
          dealii::QIterated<1>(dealii::QMidpoint<1>(), 4 * num_azim)) {};

template class QPglc<1>;
template class QPglc<2>;
template class QPglc<3>;

}  // namespace aether::sn