#ifndef AETHER_SN_TRANSPORT_H_
#define AETHER_SN_TRANSPORT_H_

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/base/iterator_range.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/loop.h>

#include "quadrature.hpp"

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
            const dealii::Quadrature<qdim> &quadrature);

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

 protected:
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

  /**
   * Assemble the cell contributions of the local matrix.
   * 
   * @param ordinates_in_sweep Ordinates in the current sweep.
   * @param fe_values Cell finite element values.
   * @param rhs_cell The right-hand-side vectors by ordinate (block).
   * @param cross_section The total material cross-section of the cell.
   * @param matrices The local (cell) matrices by ordinate.
   * @param src_cell The source vectors by ordinate (block).
   */
  void integrate_cell_term(const std::vector<Ordinate> &ordinates_in_sweep,
                           const dealii::FEValues<dim> &fe_values,
                           const dealii::BlockVector<double> &rhs_cell,
                           double cross_section,
                           std::vector<dealii::FullMatrix<double>> &matrices,
                           dealii::BlockVector<double> &src_cell) const;

  /**
   * Assemble the boundary contributions of the local matrix.
   * 
   * @param ordinates_in_sweep Ordinates in the current sweep.
   * @param fe_face_values Face finite element values.
   * @param dst_boundary The values of \f$\psi_\text{inc}\f$ on the face.
   * @param matrices The local (cell) matrices by ordinate.
   * @param src_cell The source vectors by ordinate (block).
   */
  void integrate_boundary_term(const std::vector<Ordinate> &ordinates_in_sweep,
                               const dealii::FEFaceValues<dim> &fe_face_values,
                               const dealii::BlockVector<double> &dst_boundary,
                               std::vector<dealii::FullMatrix<double>> &matrices,
                               dealii::BlockVector<double> &src_cell) const;

  /**
   * Assemble the face contributions of the local matrix.
   * 
   * @param ordinates_in_sweep Ordinates in the current sweep.
   * @param fe_face_values Face finite element values.
   * @param fe_face_values_neighbor Face finite element values of neighbor.
   * @param dst_boundary The values of \f$\psi\f$ in the neighboring cell.
   * @param matrices The local (cell) matrices by ordinate.
   * @param src_cell The source vectors by ordinate (block).
   */
  void integrate_face_term(
      const std::vector<Ordinate> &ordinates_in_sweep,
      const dealii::FEFaceValuesBase<dim> &fe_face_values,
      const dealii::FEFaceValuesBase<dim> &fe_face_values_neighbor,
      const dealii::BlockVector<double> &dst_neighbor,
      std::vector<dealii::FullMatrix<double>> &matrices,
      dealii::BlockVector<double> &src_cell) const;

  //! DoF handler for the finite elelments.
  const dealii::DoFHandler<dim> &dof_handler;
  //! Angular quadrature.
  const dealii::Quadrature<qdim> &quadrature;
  //! Discrete ordinates (sweep directions).
  std::vector<Ordinate> ordinates;
  //! Representative direction per unique octant of unit sphere.
  std::vector<Ordinate> octant_directions;
  //! Downstream ordering of cells, by unique octant of unit sphere.
  std::vector<std::vector<ActiveCell>> cells_downstream;
  //! Map of octant ordinate indices to global ordinate indices.
  std::vector<std::vector<int>> octants_to_global;
};

#endif  // AETHER_SN_TRANSPORT_H_