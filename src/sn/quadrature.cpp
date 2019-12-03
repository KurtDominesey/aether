#include "quadrature.hpp"

dealii::Quadrature<2> gauss_chebyshev(int order) {
  int num_points = 2 * order - 1;
  dealii::QGauss<1> polar(num_points);
  dealii::QGaussChebyshev<1> azimuthal(num_points);
  return dealii::Quadrature<2>(polar, azimuthal);
}

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

dealii::Quadrature<2> impose_polar_symmetry(
    const dealii::Quadrature<2> &quadrature) {
  const std::vector<dealii::Point<2>> &points = quadrature.get_points();
  const std::vector<double> &weights = quadrature.get_weights();
  std::vector<dealii::Point<2>> points_sym;
  std::vector<double> weights_sym;
  points_sym.reserve(points.size()/2);
  weights_sym.reserve(weights.size()/2);
  for (int n = 0; n < points.size(); ++n) {
    bool both_pos = points[n](0) > 0.5 && points[n](1) > 0.5;
    bool both_neg = points[n](0) < 0.5 && points[n](1) < 0.5;
    if (both_pos || both_neg) {
      points_sym.push_back(points[n]);
      weights_sym.push_back(weights[n]*2);
    }
  }
  dealii::Quadrature<2> quadrature_sym(points_sym, weights_sym);
  return quadrature_sym;
}

template <int dim>
dealii::Quadrature<dim> reorder(const dealii::Quadrature<dim> &quadrature) {
  const std::vector<dealii::Point<dim>> &points = quadrature.get_points();
  const std::vector<double> &weights = quadrature.get_weights();
  std::vector<int> indices(points.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(), CompareQuadraturePoints(points));
  std::vector<dealii::Point<dim>> points_sorted(points.size());
  std::vector<double> weights_sorted(weights.size());
  for (int i = 0; i < points.size(); ++i) {
    points_sorted[indices[i]] = points[i];
    weights_sorted[indices[i]] = weights[i];
  }
  return dealii::Quadrature<dim>(points_sorted, weights_sorted);
}

template dealii::Quadrature<1> reorder(const dealii::Quadrature<1>&);
template dealii::Quadrature<2> reorder(const dealii::Quadrature<2>&);

template dealii::Tensor<1, 1> ordinate(const dealii::Point<1>);
template dealii::Tensor<1, 2> ordinate(const dealii::Point<2>);
template dealii::Tensor<1, 3> ordinate(const dealii::Point<2>);