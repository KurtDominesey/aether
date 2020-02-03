#include "quadrature.h"

namespace aether::sn {

template <int dim, int qdim>
dealii::Tensor<1, dim> ordinate(const dealii::Point<qdim> coordinate) {
  dealii::Point<dim> ordinate;
  double cos_theta = 2 * coordinate(0) - 1;
  if (qdim == 1) {
    Assert(dim == 1, dealii::ExcInvalidState());
    ordinate(0) = cos_theta;
  }
  else {
    Assert(qdim == 2, dealii::ExcInvalidState());
    Assert(2 <= dim && dim <= 3, dealii::ExcInvalidState());
    double phi = coordinate(1) * 2 * dealii::numbers::PI;
    double polar_proj = std::sqrt(1 - std::pow(cos_theta, 2));
    ordinate(0) = polar_proj * std::cos(phi);
    ordinate(1) = polar_proj * std::sin(phi);
    if (dim == 3)
      ordinate(2) = cos_theta;
  }
  return ordinate;
}

template dealii::Tensor<1, 1> ordinate(const dealii::Point<1>);
template dealii::Tensor<1, 2> ordinate(const dealii::Point<2>);
template dealii::Tensor<1, 3> ordinate(const dealii::Point<2>);

dealii::Quadrature<1> impose_polar_symmetry(
    const dealii::Quadrature<1> &quadrature) {
  auto points = quadrature.get_points();
  auto weights = quadrature.get_weights();
  for (int n = 0, nn = 0; nn < quadrature.size(); ++n, ++nn) {
    if (points[n][0] > 0.5) {
      weights[n] *= 2;
      continue;
    }
    points.erase(points.begin() + n);
    weights.erase(weights.begin() + n);
    --n;
  }
  return dealii::Quadrature<1>(points, weights);
}

template <int dim, int qdim>
QAngle<dim, qdim>::QAngle(const unsigned int num_quadrature_points)
    : dealii::Quadrature<qdim>(num_quadrature_points) {
  q_angle_init();
}

template <int dim, int qdim>
QAngle<dim, qdim>::QAngle(
    const typename dealii::Quadrature<qdim>::SubQuadrature &sub_quadrature,
    const dealii::Quadrature<1>& quadrature)
    : dealii::Quadrature<qdim>(sub_quadrature, quadrature) {
  q_angle_init();
}

template <int dim, int qdim>
QAngle<dim, qdim>::QAngle(
    const dealii::Quadrature<qdim != 1 ? 1 : 0> &quadrature_1d)
    : dealii::Quadrature<qdim>(quadrature_1d) {
  q_angle_init();
}

template <int dim, int qdim>
QAngle<dim, qdim>::QAngle(const dealii::Quadrature<qdim> &quadrature)
    : dealii::Quadrature<qdim>(quadrature) {
  q_angle_init();
}

template <int dim, int qdim>
QAngle<dim, qdim>::QAngle(const std::vector<dealii::Point<qdim>> &points,
                          const std::vector<double> &weights)
    : dealii::Quadrature<qdim>(points, weights) {
  q_angle_init();
}

template <int dim, int qdim>
QAngle<dim, qdim>::QAngle(const std::vector<dealii::Point<qdim>> &points)
    : dealii::Quadrature<qdim>(points) {
  q_angle_init();
}

template <int dim, int qdim>
QAngle<dim, qdim>::QAngle(const dealii::Point<qdim> &point)
    : dealii::Quadrature<qdim>(point) {
  q_angle_init();
}

template <int dim, int qdim>
QAngle<dim, qdim>& QAngle<dim, qdim>::operator=(
    const QAngle<dim, qdim> &other) {
  dealii::Quadrature<qdim>::operator=(other);
  quadrature_angles = other.quadrature_angles;
  ordinates = other.ordinates;
}

template <int dim, int qdim>
bool QAngle<dim, qdim>::operator==(const QAngle<dim, qdim> &other) const {
  bool base = dealii::Quadrature<qdim>::operator==(other);
  return base && (quadrature_angles == other.quadrature_angles) &&
                 (ordinates == other.ordinates);
}

template <int dim, int qdim>
void QAngle<dim, qdim>::initialize(
    const std::vector<dealii::Point<qdim>> &points, 
    const std::vector<double> &weights) {
  dealii::Quadrature<qdim>::initialize(points, weights);
  q_angle_init();
}

template <int dim, int qdim>
const dealii::Point<qdim>& QAngle<dim, qdim>::angle(const int n) const {
  AssertIndexRange(n, this->size());
  return quadrature_angles[n];
}

template <int dim, int qdim>
const std::vector<dealii::Point<qdim>>& QAngle<dim, qdim>::get_angles() const {
  return quadrature_angles;
}

template <int dim, int qdim>
const dealii::Tensor<1, dim>& QAngle<dim, qdim>::ordinate(const int n) const {
  AssertIndexRange(n, this->size());
  return ordinates[n];
}

template <int dim, int qdim>
const std::vector<dealii::Tensor<1, dim>> &QAngle<dim, qdim>::get_ordinates()
    const {
  return ordinates;
}

template <int dim, int qdim>
void QAngle<dim, qdim>::q_angle_init() {
  if (dim == 2) {  // impose polar symmetry
    const int size_original = this->size();
    for (int n = 0, nn = 0; n < size_original; ++n, ++nn) {
      if (this->quadrature_points[nn][0] > 0.5)
        continue;
      this->quadrature_points.erase(this->quadrature_points.begin() + nn);
      this->weights.erase(this->weights.begin() + nn);
      --nn;
    }
    for (int n = 0; n < this->size(); ++n)
      this->weights[n] *= 2;
    Assert(this->size() == size_original / 2, dealii::ExcImpossibleInDim(dim));
  }
  quadrature_angles = this->quadrature_points;
  for (int n = 0; n < quadrature_angles.size(); ++n) {
    quadrature_angles[n][0] *= 2;
    quadrature_angles[n][0] -= 1;
    if (qdim == 2)
      quadrature_angles[n][1] *= 2 * dealii::numbers::PI;
  }
  ordinates.resize(quadrature_angles.size(), dealii::Point<dim>());
  for (int n = 0; n < quadrature_angles.size(); ++n) {
    const double polar = quadrature_angles[n][0];
    if (dim == 1)
      ordinates[n][0] = polar;
    else {
      double proj = std::sqrt(1 - std::pow(polar, 2));
      ordinates[n][0] = proj * std::cos(quadrature_angles[n][1]);
      ordinates[n][1] = proj * std::sin(quadrature_angles[n][1]);
      if (dim == 3)
        ordinates[n][2] = polar;
    }
  }
}

template <int dim, int qdim>
int QAngle<dim, qdim>::reflected_index(
    const int n, const dealii::Tensor<1, dim> &normal) const {
  throw dealii::ExcMessage(
      "This angular quadrature does not support refecting boundaries");
}

template class QAngle<1>;
template class QAngle<2>;
template class QAngle<3>;

}  // namespace aether::sn