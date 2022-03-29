#include "quadrature_lib.h"

namespace aether::sn {

/**
 * Constructor for 1D.
 * 
 * Throws an error if a nonzero value of `num_azim` is specified.
 */
template <>
QPglc<1, 1>::QPglc(const int num_polar, const int num_azim)
    : QAngle<1, 1>(dealii::QGauss<1>(2 * num_polar)) {
  Assert(num_azim == 0, dealii::ExcImpossibleInDim(1));
}

template <int dim, int qdim>
QPglc<dim, qdim>::QPglc(const int num_polar, const int num_azim)
    : QAngle<dim, qdim>(
          dealii::QGauss<1>(2 * num_polar),
          dealii::QIterated<1>(dealii::QMidpoint<1>(), 4 * num_azim)) {}

template <int dim, int qdim>
int QPglc<dim, qdim>::reflected_index(
    const int n, const dealii::Tensor<1, dim> &normal) const {
  int n_refl = -1;
  const dealii::Tensor<1, dim> &ordinate = this->ordinate(n);
  const dealii::Tensor<1, dim> ordinate_refl = 
      ordinate - 2 * (ordinate * normal) * normal;
  if (qdim == 1) {
    n_refl = this->size() - 1 - n;
  } else {
    Assert(qdim == 2, dealii::ExcInvalidState());
    Assert(this->is_tensor_product(), dealii::ExcInvalidState());
    const dealii::Quadrature<1> &q_polar = this->get_tensor_basis()[0];
    const dealii::Quadrature<1> &q_azim = this->get_tensor_basis()[1];
    int n_polar = n % q_polar.size();
    int n_azim  = n / q_polar.size();
    Assert(n == (n_azim * q_polar.size() + n_polar), dealii::ExcInvalidState());
    // only works for extruded geometry in 3D
    int n_polar_refl = n_polar;
    if (dim == 3 && ordinate_refl[2] == -ordinate[2])
      n_polar_refl = q_polar.size() - 1 - n_polar;
    double polar_refl = 2 * q_polar.point(n_polar)[0] - 1;
    double proj = std::sqrt(1-std::pow(polar_refl, 2));
    // acos returns azimuthal angle in [0, pi], upper half of unit disk
    double azim_refl = std::acos(ordinate_refl[0]/proj);
    if (ordinate_refl[1] < 0)  // y < 0, lower half of unit disk
      azim_refl = 2 * dealii::numbers::PI - azim_refl;
    double azim = 2 * dealii::numbers::PI * q_azim.point(n_azim)[0];
    double azim_rotate = azim_refl - azim;
    double sector = 2 * dealii::numbers::PI / (double)q_azim.size();
    int n_rotate = std::round(azim_rotate/sector);
    Assert(std::abs(azim_rotate/sector - n_rotate) < 1e-12,
           dealii::ExcNotMultiple(azim_rotate, sector));
    int n_azim_refl = n_azim + n_rotate;
    // index wraps around if it goes out of bounds
    if (n_azim_refl < 0)
      n_azim_refl = q_azim.size() - n_azim_refl;
    else if (n_azim_refl > q_azim.size())
      n_azim_refl = n_azim_refl % q_azim.size();
    n_refl = n_azim_refl * q_polar.size() + n_polar_refl;
  }
  // ordinates are only of unit norm in 3D
  double error = this->ordinate(n_refl) * ordinate_refl
                  - this->ordinate(n_refl).norm() * ordinate_refl.norm();
  AssertThrow(std::abs(error) < 1e-12, 
              dealii::ExcMessage("Ordinate not reflecting"));
  return n_refl;
}

template class QPglc<1>;
template class QPglc<2>;
template class QPglc<3>;

QForwardBackward::QForwardBackward() : QAngle(
    std::vector<dealii::Point<1>>{dealii::Point<1>(-1), dealii::Point<1>(+1)},
    std::vector<double>{0.5, 0.5}) {}

int QForwardBackward::reflected_index(
    const int n, const dealii::Tensor<1, 1> &normal) const {
  return !n;
}

}  // namespace aether::sn