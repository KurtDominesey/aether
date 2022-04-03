#ifndef AETHER_SN_QUADRATURE_H_
#define AETHER_SN_QUADRATURE_H_

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/point.h>

namespace aether::sn {

template <int dim, int qdim>
dealii::Tensor<1, dim> ordinate(const dealii::Point<qdim> coordinate);

dealii::Quadrature<1> impose_polar_symmetry(
    const dealii::Quadrature<1> &quadrature);

/**
 * Angular quadrature.
 * 
 * Normalized to unity, not \f$4\pi\f$.
 */
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
  /**
   * Return the `n`th angle.
   * 
   * Angles are stored as \f$(\mu,\omega)\f$ where \f$\mu\in[-1,1]\f$ is the
   * cosine of the polar angle and \f$\omega\in[0,2\pi]\f$ is the azimuthal
   * angle.  
   */
  const dealii::Point<qdim>& angle(const int n) const;
  /**
   * Returns a reference to the whole vector of angles.
   */
  const std::vector<dealii::Point<qdim>>& get_angles() const;
  /**
   * Returns the `n`th ordinate.
   */
  const dealii::Tensor<1, dim>& ordinate(const int n) const;
  /**
   * Returns a reference to the whole vector of ordinates.
   */
  const std::vector<dealii::Tensor<1, dim>>& get_ordinates() const;
  /**
   * Returns the index corresponding to the reflection of a given ordinate.
   * 
   * Throws an error if the reflected ordinate is not in the quadrature.
   * 
   * @param n Index of the incident ordinate
   * @param normal Normal vector of the reflecting surface
   */
  virtual int reflected_index(
      const int n, const dealii::Tensor<1, dim> &normal) const;
   /**
    * Whether the quadrature lacks meaningful polar angles.
    * 
    * For `qdim` == 1, this means there are two polar angles, {$\pm 1,\mp 1$}.
    * For `qdim` == 2, this means all polar angles are zero.
    */
   bool is_degenerate() const;
 protected:
  //! Vector of quadrature angles.
  std::vector<dealii::Point<qdim>> quadrature_angles;
  //! Vector of ordinates corresponding to the quadrature angles.
  std::vector<dealii::Tensor<1, dim>> ordinates;
  /**
   * Initializes @ref QAngle::quadrature_angles and @ref QAngle::ordinates 
   * and asserts polar symmetry if `dim==2`. 
   * 
   * To be called after quadrature points and weights have been set.
   */
  void q_angle_init();
  //! Whether the quadrature lacks meaningful polar angles.
  bool _is_degenerate = true;
};

template <int dim, int qdim>
inline bool QAngle<dim, qdim>::is_degenerate() const {
  return _is_degenerate;
}

}  // namespace aether::sn

#endif  // AETHER_SN_QUADRATURE_H_