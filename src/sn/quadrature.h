#ifndef AETHER_SN_QUADRATURE_H_
#define AETHER_SN_QUADRATURE_H_

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/point.h>

namespace aether::sn {

template <int dim, int qdim>
dealii::Tensor<1, dim> ordinate(const dealii::Point<qdim> coordinate);

dealii::Quadrature<1> impose_polar_symmetry(
    const dealii::Quadrature<1> &quadrature);

template <int dim, int qdim = dim == 1 ? 1 : 2>
class QAngle : public dealii::Quadrature<qdim> {
 public:
  explicit QAngle(const unsigned int num_quadrature_points = 0);
  QAngle(const typename dealii::Quadrature<qdim>::SubQuadrature &sub_quadrature, 
         const dealii::Quadrature<1> &quadrature);
  explicit QAngle(const dealii::Quadrature<qdim != 1 ? 1 : 0> &quadrature_1d);
  QAngle(const dealii::Quadrature<qdim> &quadrature);
  QAngle(QAngle<dim, qdim>&&) noexcept = default;
  QAngle(const std::vector<dealii::Point<qdim>> &points,
         const std::vector<double> &weights);
  QAngle(const std::vector<dealii::Point<qdim>> &points);
  QAngle(const dealii::Point<qdim> &point);
  QAngle<dim, qdim>& operator=(const QAngle<dim, qdim> &other);
  QAngle<dim, qdim>& operator=(QAngle<dim, qdim>&&) = default;
  bool operator==(const QAngle<dim, qdim> &other) const;
  void initialize(const std::vector<dealii::Point<qdim>> &points, 
                  const std::vector<double> &weights);
  const dealii::Point<qdim>& angle(const int n) const;
  const std::vector<dealii::Point<qdim>>& get_angles() const;
  const dealii::Tensor<1, dim>& ordinate(const int n) const;
  const std::vector<dealii::Tensor<1, dim>>& get_ordinates() const;
  int reflected_index(const int n, const dealii::Tensor<1, dim> &normal) const;
 protected:
  std::vector<dealii::Point<qdim>> quadrature_angles;
  std::vector<dealii::Tensor<1, dim>> ordinates;
  void q_angle_init();
};

}  // namespace aether::sn

#endif  // AETHER_SN_QUADRATURE_H_