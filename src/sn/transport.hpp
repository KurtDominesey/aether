#ifndef AETHER_SN_TRANSPORT_H_
#define AETHER_SN_TRANSPORT_H_

#include <deal.II/lac/block_vector.h>

/**
 * A matrix-free expression of the linear operator
 * \f$L\psi=\right(\Omega\cdot\nabla+\Sigma_t\right)\psi\f$.
 */
template <int dim, int qdim = dim == 1 ? 1 : 2>
class Sweeper {
  using Ordinate = dealii::Tensor<1, dim>;

 public:
  /**
   * Constructor.
   * 
   * @param dof_handler DoF handler for finite elements.
   * @param quadrature Angular quadrature.
   * @param cross_sections Total material cross-sections.
   */
  Sweeper(const dealii::DoFHandler<dim> &dof_handler,
          const dealii::Quadrature<qdim> &quadrature,
          const std::vector<double> &cross_sections);
  /**
   * Compute \f$L^{-1}q\f$.
   * 
   * @param dst Destination vector.
   * @param src Source vector (\f$q\f$).
   */
  void sweep(dealii::BlockVector<double> &dst,
             dealii::BlockVector<double> &src);
  /**
   * Compute \f$L^{-1}q\f$ for a single octant of the unit sphere.
   * 
   * @param dst Destination vector (one octant).
   * @param src Source vector (\f$q\f$).
   */
  void sweep_octant(dealii::BlockVector<double> &dst,
                    dealii::BlockVector<double> &src);
  const dealii::DoFHandler<dim> &dof_handler;
  const dealii::Quadrature<qdim> &quadrature;
  std::vector<double> &cross_sections;
  std::vector<Ordinate> octant_directions;

 protected:
  using DoFInfo = dealii::MeshWorker::DoFInfo<dim>;
  using CellInfo = dealii::MeshWorker::IntegrationInfo<dim>;
  static void integrate_cell_term(DoFInfo &dinfo, CellInfo &info);
  static void integrate_boundary_term(DoFInfo &dinfo, CellInfo &info);
  static void integrate_face_term(DoFInfo &dinfo, CellInfo &info);
};

template <int dim, int qdim>
Sweeper<dim, qdim>::Sweeper(const dealii::DoFHandler<dim> &dof_handler,
                            const dealii::Quadrature<qdim> &quadrature) 
    : dof_handler(dof_handler), quadrature(quadrature) {
  octant_directions.resize(std::pow(dim, 2));
  if (dim == 1) {
    octant_directions[0] = dealii::Point<dim>(+1);
    octant_directions[1] = dealii::Point<dim>(-1);
  } else if (dim == 2) {
    octant_directions[0] = dealii::Point<dim>(+1, +1);
    octant_directions[1] = dealii::Point<dim>(-1, +1);
    octant_directions[2] = dealii::Point<dim>(-1, -1);
    octant_directions[3] = dealii::Point<dim>(+1, -1);
  } else if (dim == 3) {
    octant_directions[0] = dealii::Point<dim>(+1, +1, +1);
    octant_directions[1] = dealii::Point<dim>(-1, +1, +1);
    octant_directions[2] = dealii::Point<dim>(-1, -1, +1);
    octant_directions[3] = dealii::Point<dim>(+1, -1, +1);
    octant_directions[4] = dealii::Point<dim>(+1, +1, -1);
    octant_directions[5] = dealii::Point<dim>(-1, +1, -1);
    octant_directions[6] = dealii::Point<dim>(-1, -1, -1);
    octant_directions[7] = dealii::Point<dim>(+1, -1, -1);
  }
}

template <int dim, int qdim>
void Sweeper<dim, qdim>::integrate_cell_term(DoFInfo &dinfo, 
                                             CellInfo &info) {
  const dealii::FEValuesBase<dim> &fe_values = info.fe_values();
  dealii::FullMatrix<double> &local_matrix = dinfo.matrix(0).matrix;
  const std::vector<double> &JxW = fe_values.get_JxW_values();
  for (int q = 0; q < fe_values.n_quadrature_points; ++q) {
    for (int i = 0; i < fe_values.dofs_per_cell; ++i) {
      for (int j = 0; j < fe_values.dofs_per_cell; ++j) {
        double streaming = 
          -direction * fe_values.shape_grad(i, q) * fe_values.shape_value(j, q);
        double collision =
          fe_values.shape_value(i, q) * fe_values.shape_value(j, q);
        local_matrix(i, j) += (streaming + collision) * JxW[q];
      }
    }
  }
}

template <int dim, int qdim>
void Sweeper<dim, qdim>::integrate_boundary_term(DoFInfo &dinfo, CellInfo &info) {

}

class SweepInverter {
 public:
  void initialize(dealii::BlockVector<double> &dst, 
                  dealii::BlockVector<double> &src);

  template <class DOFINFO>
  void initialize_info(DOFINFO &info, bool face) const {
    info.initialize_matrices(dst.n_blocks(), face);
    info.initialize_vectors(dst.n_blocks());
  };

  template <class DOFINFO>
  void assemble(const DOFINFO &info) {
    for (int ordinate = 0; ordinate < info.n_matrices(); ++ordinate) {
      dealii::FullMatrix<double> &matrix = info.matrix(ordinate).matrix;
      matrix.gauss_jordan();  // directly invert
      matrix.vmult(dst.block(ordinate), src.block(ordinate));
    }
  };

  template <class DOFINFO>
  void assemble(const DOFINFO &info1, const DOFINFO &info2) {};

 protected:
  dealii::BlockVector<double> &dst;
  dealii::BlockVector<double> &src;
};



#endif  // AETHER_SN_TRANSPORT_H_