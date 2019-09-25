#ifndef AETHER_SN_CONSERVATION_H_
#define AETHER_SN_CONSERVATION_H_

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/fe_dgq.h>

#include "scattering.hpp"
#include "discrete_to_moment.hpp"
#include "moment_to_discrete.hpp"

/**
 * A statement of particle conservation for radiation transport.
 * 
 * Expresses the first-order, linear radiation transport equation with 
 * discontinuous Galerkin finite elements in space and an S<sub>N</sub> 
 * (quadrature) discretization in angle.
 */
template <int dim, int qdim = dim == 1 ? 1 : 2>
class Conservation {
 using Ordinate = dealii::Tensor<1, dim>;
 template <typename T>
 using Separated = std::map<char, std::vector<T> >;

 public:
  /**
   * Constructor.
   * 
   * @param fe The discontinuous Galerkin finite elements in space.
   * @param quadrature The angular quadrature.
   * @param mesh Meshed problem domain.
   */
  Conservation(dealii::FE_DGQ<dim> &fe, dealii::Quadrature<qdim> &quadrature,
               dealii::Triangulation<dim> &mesh, 
               Separated<double> &cross_sections);
  /**
   * Apply the linear operator.
   * 
   * @param dst Destination vector.
   * @param src Source vector.
   */
  void vmult(dealii::BlockVector<double> &dst,
             const dealii::BlockVector<double> &src) const;
  /**
   * Apply the transpose of the linear operator (not implemented)
   * 
   * @param dst Destination vector.
   * @param src Source vector.
   */
  void Tvmult(dealii::BlockVector<double> &dst,
              const dealii::BlockVector<double> &src) const;
  // dealii::BlockVector<double> solution;

 protected:
  dealii::FE_DGQ<dim> &fe;
  dealii::Quadrature<qdim> &quadrature;
  dealii::Triangulation<dim> &mesh;
  Separated<double> &cross_sections;
  // MatrixType transport;
  // std::vector<dealii::Function<dim> *> sources;
  // std::vector<dealii::BlockVector<double> > sources_h;

 public:
  dealii::DoFHandler<dim> dof_handler;
  dealii::QGauss<dim> quadrature_fe;
  DiscreteToMoment<qdim> d2m;
  Scattering<dim> scattering;
  MomentToDiscrete<qdim> m2d;
  // std::vector<dealii::Point<qdim> > angles;
  // std::vector<Ordinate> ordinates;
};

template <int dim, int qdim>
Conservation<dim, qdim>::Conservation(
    dealii::FE_DGQ<dim> &fe, dealii::Quadrature<qdim> &quadrature,
    dealii::Triangulation<dim> &mesh, Separated<double> &cross_sections)
    : fe(fe),
      quadrature(quadrature),
      mesh(mesh),
      cross_sections(cross_sections),
      dof_handler(mesh),
      quadrature_fe(fe.get_degree()+1),
      d2m(quadrature),
      scattering(dof_handler, cross_sections['S']),
      m2d(quadrature) {
  dof_handler.distribute_dofs(fe);
  // int num_points = quadrature.get_points().size();
  // angles.reserve(num_points);
  // ordinates.reserve(num_points);
  // for (int p = 0; p < num_points; ++p) {
  //   dealii::Point<qdim> cos_point = quadrature.point(p);
  //   angles.emplace_back(q_point_to_angle);
  //   ordinates.emplace_back(q_point_to_ordinate(point));
  // }
}

template <int dim, int qdim>
void Conservation<dim, qdim>::vmult(dealii::BlockVector<double> &flux,
                                    const dealii::BlockVector<double> &source) 
                                    const {
  int num_ordinates = quadrature.get_weights().size();
  dealii::BlockVector<double> discrete(num_ordinates, dof_handler.n_dofs());
  m2d.vmult(discrete, source);
  d2m.vmult(flux, discrete);
  // d2m.vmult(source_m, source_d)
  // scattering.vmult(flux, source);
  // m2d.vmult(scattered_d, scattered_m)
  // transport.vmult(dst, scattered_d);
  // auto I = identity
  // flux = (I - D2M * L_inv * M2D * S) * D2M * source;
  // auto L_inv = transport.linear_operator(coeffs_extrinsic);
  // auto S = scattering.linear_operator(coeffs_extrinsic);
  // auto I = identity;
  // auto inverse = inverse_operator(I-S*L);
  // dst = inverse * src;
}

template <int dim, int qdim>
void Conservation<dim, qdim>::Tvmult(dealii::BlockVector<double> &dst,
                                     const dealii::BlockVector<double> &src) 
                                     const {
}

// template <int dim, int qdim>
// LinearOperator<> Conservation<dim, qdim>::linear_operator(
//     Seprated<double> coeffs_extrinsic) {
//   coeffs_extrinsic = coeffs_extrinsic;
//   return linear_operator(*this);
// }

#endif  // AETHER_SN_CONSERVATION_H_ 