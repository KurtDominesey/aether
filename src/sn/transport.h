#ifndef AETHER_SN_TRANSPORT_H_
#define AETHER_SN_TRANSPORT_H_

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>

#include "quadrature.h"
#include "types/types.h"
#include "base/mgxs.h"

namespace aether::pgd::sn {
template <int dim, int qdim> class FixedSourceP;
}

namespace aether::sn {
template <int dim, int qdim> class Transport;
}

namespace aether::sn {

template <int dim>
struct CellMatrices {
  CellMatrices(int num_dofs, int num_faces, int num_q_points)
      : mass(num_dofs),
        grad(num_dofs, num_dofs),
        normals(num_faces, num_q_points),
        outflow(num_faces, num_dofs, num_dofs),
        inflow(num_faces) {};
  dealii::FullMatrix<double> mass;
  dealii::Table<2, dealii::Tensor<1, dim>> grad;
  dealii::Table<2, dealii::Tensor<1, dim>> normals;
  dealii::Table<3, dealii::Tensor<1, dim>> outflow;
  std::vector<std::vector<dealii::Table<2, dealii::Tensor<1, dim>>>> inflow;
};

/**
 * A matrix-free expression of the linear operator \f$L^{-1}\f$ where
 * \f$L\psi=\left(\Omega\cdot\nabla+\Sigma_t\right)\psi\f$.
 *
 * Specifically, discretizing \f$\Omega\f$ to discrete ordinates 
 * \f$\Omega_1,\ldots\Omega_N\f$ we have 
 * \f[
 *   \left(\Omega_n\cdot\nabla+\Sigma_t(r)\right)\psi_n(r)=q(r)\quad
 *   \forall r\in\mathcal{V},\ n\in[0,N]
 * \f]
 * with boundary conditions
 * \f[
 *   \psi_n(r)=\psi_{inc}\quad \forall r\in\Gamma_-
 * \f]
 * where \f$\Gamma_-\f$ is the inflow boundary,
 * \f[
 *   \Gamma_-:=\{r\in\partial\mathcal{V},\ \Omega_n\cdot n(r)<0\}.
 * \f]
 * We achieve a variational form by multiplying by test function \f$\psi^*\f$
 * and integrating over the problem domain \f$\mathcal{V}\f$. Subsequently, we
 * integrate by parts. Inspecting a single element \f$K\f$ (and suppressing
 * subscript \f$n\f$),
 * \f[
 *   \left(\psi,-\Omega\cdot\nabla\psi^*\right)_K
 *   + \left\langle\hat{\psi},\Omega\cdot n\psi^*_-\right\rangle_{\partial K_+}
 *   - \left\langle\hat{\psi},\Omega\cdot n\psi^*_+\right\rangle_{\partial K_-}
 *   + \left(\Sigma_t\psi,\psi^*\right)_K
 *   = \left(q,\psi^*\right)_K
 * \f]
 * where \f$\psi\f$ and \f$\psi^*\f$ can be discontinuous across faces. As such,
 * we use subscripts to denote the upwind and downwind values, respectively
 * \f[
 *   \psi^*_+=\lim_{s\rightarrow 0^+} \psi^*(r+s\Omega) \\
 *   \psi^*_-=\lim_{s\rightarrow 0^-} \psi^*(r+s\Omega).
 * \f]
 * We achieve closure by applying the upwind conditions on the numerical flux,
 * \f[
 *  \hat{\psi}=\psi^-\in\partial K\setminus\Gamma_- \\
 *  \hat{\psi}=\psi_{inc} \in\Gamma_-
 * \f]
 * TODO: finish documentation
 */
template <int dim, int qdim = dim == 1 ? 1 : 2>
class Transport {
 public:
  using Ordinate = dealii::Tensor<1, dim>;
  using ActiveCell = typename dealii::DoFHandler<dim>::active_cell_iterator;
  using Cell = typename dealii::DoFHandler<dim>::cell_iterator;
  using Face = typename dealii::DoFHandler<dim>::face_iterator;

  /**
   * Constructor.
   * 
   * @param dof_handler DoF handler for finite elements.
   * @param quadrature Angular quadrature.
   */
  Transport(const dealii::DoFHandler<dim> &dof_handler,
            const QAngle<dim, qdim> &quadrature);

  virtual ~Transport() {}

  /**
   * Compute \f$L^{-1}q\f$.
   * 
   * @param dst Destination vector.
   * @param src Source vector (\f$q\f$).
   * @param cross_sections Total material cross_sections.
   * @param boundary_conditions Values of \f$\psi_\text{inc}\f$ by boundary id.
   */
  void vmult(dealii::Vector<double> &dst, 
             const dealii::Vector<double> &src,
             const std::vector<double> &cross_sections,
             const std::vector<dealii::BlockVector<double>>
                 &boundary_conditions) const;

  /**
   * Compute \f$L^{-1}q\f$.
   * 
   * @param dst Destination vector.
   * @param src Source vector (\f$q\f$).
   * @param cross_sections Total material cross_sections.
   * @param boundary_conditions Values of \f$\psi_\text{inc}\f$ by boundary id.
   */
  void vmult(dealii::BlockVector<double> &dst,
             const dealii::BlockVector<double> &src,
             const std::vector<double> &cross_sections,
             const std::vector<dealii::BlockVector<double>> 
                 &boundary_conditions) const;
  
  /**
   * Return the number of blocks in a column.
   */
  int n_block_cols() const;

  /**
   * Return the number of blocks in a row.
   */ 
  int n_block_rows() const;

  dealii::BlockIndices get_block_indices() const;

  //! Cached cell matrices
  std::vector<CellMatrices<dim>> cell_matrices;
  //! DoF handler for the finite elelments.
  const dealii::DoFHandler<dim> &dof_handler;
  //! Angular quadrature.
  const QAngle<dim, qdim> &quadrature;

 protected:
  void assemble_cell_matrices();

  /**
   * Compute \f$L^{-1}q\f$ for a single octant of the unit sphere.
   * 
   * @param oct The index of the octant.
   * @param dst Destination vector (one octant).
   * @param src Source vector (\f$q\f$).
   * @param cross_sections Total material cross_sections.
   * @param boundary_conditions Values of \f$\psi_\text{inc}\f$ by boundary id.
   */
  void vmult_octant(int oct, dealii::BlockVector<double> &dst,
                    const dealii::BlockVector<double> &src,
                    const std::vector<double> &cross_sections,
                    const std::vector<dealii::BlockVector<double>>
                        &boundary_conditions) const;

  //! Discrete ordinates (sweep directions).
  std::vector<Ordinate> ordinates;
  //! Representative direction per unique octant of unit sphere.
  std::vector<Ordinate> octant_directions;
  //! Active DoF cells in z-order
  std::vector<ActiveCell> cells;
  //! Downstream ordering of cells, by ordinate.
  std::vector<std::vector<int>> sweep_orders;
  //! Map of octant ordinate indices to global ordinate indices.
  std::vector<std::vector<int>> octants_to_global;

  friend class aether::pgd::sn::FixedSourceP<dim, qdim>;
};

}  // namespace aether::sn

#endif  // AETHER_SN_TRANSPORT_H_