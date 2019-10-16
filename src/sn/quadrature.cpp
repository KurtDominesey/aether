#include "quadrature.hpp"

dealii::Quadrature<2> gauss_chebyshev(int order) {
  int num_points = 2 * order - 1;
  dealii::QGauss<1> polar(num_points);
  dealii::QGaussChebyshev<1> azimuthal(num_points);
  return dealii::Quadrature<2>(polar, azimuthal);
}

dealii::Quadrature<2> impose_azimuthal_symmetry(
    const dealii::Quadrature<2> &quadrature) {
  const std::vector<dealii::Point<2> > &points = quadrature.get_points();
  const std::vector<double> &weights = quadrature.get_weights();
  std::vector<dealii::Point<2> > points_sym;
  std::vector<double> weights_sym;
  points_sym.reserve(points.size()/2);
  weights_sym.reserve(weights.size()/2);
  for (int n = 0; n < points.size(); ++n) {
    if (points[n](1) > 0.5) {
      points_sym.push_back(points[n]);
      weights_sym.push_back(weights[n]*2);
    }
  }
  dealii::Quadrature<2> quadrature_sym(points_sym, weights_sym);
  return quadrature_sym;
}