#ifndef AETHER_SN_TRANSPORT_H_
#define AETHER_SN_TRANSPORT_H_

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/base/iterator_range.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/loop.h>

#include "sweeper.hpp"

/**
 * A matrix-free expression of the linear operator
 * \f$L\psi=\right(\Omega\cdot\nabla+\Sigma_t\right)\psi\f$.
 */
template <int dim, int qdim = dim == 1 ? 1 : 2>
class Transport {
  using Ordinate = dealii::Tensor<1, dim>;

 public:
  /**
   * Constructor.
   * 
   * @param dof_handler DoF handler for finite elements.
   * @param quadrature Angular quadrature.
   * @param cross_sections Total material cross-sections.
   */
  Transport(dealii::DoFHandler<dim> &dof_handler,
            const dealii::Quadrature<qdim> &quadrature,
            const std::vector<double> &cross_sections);
  /**
   * Compute \f$L^{-1}q\f$.
   * 
   * @param dst Destination vector.
   * @param src Source vector (\f$q\f$).
   */
  void vmult(dealii::BlockVector<double> &dst,
             const dealii::BlockVector<double> &src);
  /**
   * Compute \f$L^{-1}q\f$ for a single octant of the unit sphere.
   * 
   * @param dst Destination vector (one octant).
   * @param src Source vector (\f$q\f$).
   */
  void sweep_octant(dealii::BlockVector<double> &dst,
                    dealii::BlockVector<double> &src);
                    
  dealii::DoFHandler<dim> &dof_handler;
  const dealii::Quadrature<qdim> &quadrature;
  const std::vector<double> &cross_sections;
  std::vector<Ordinate> ordinates;
  std::vector<Ordinate> octant_directions;
  std::vector<dealii::types::global_dof_index> numbering;
  std::vector<std::vector<dealii::types::global_dof_index > > renumberings;

 protected:
  using DoFInfo = dealii::MeshWorker::DoFInfo<dim>;
  using CellInfo = dealii::MeshWorker::IntegrationInfo<dim>;
  /**
   * Assemble the cell contributions of the local matrix.
   * 
   * @param dinfo DoF info.
   * @param info Cell integration info.
   */
  void integrate_cell_term(std::vector<int> &octant_to_global, 
                           DoFInfo &dinfo, CellInfo &info);
  /**
   * Assemble the boundary contributions of the local matrix.
   * 
   * @param dinfo DoF info.
   * @param info Cell integration info.
   */
  void integrate_boundary_term(std::vector<int> &octant_to_global, 
                               DoFInfo &dinfo, CellInfo &info);
  /**
   * Assemble the face contributions of the local matrix.
   * 
   * @param dinfo DoF info.
   * @param info Cell integration info.
   */
  void integrate_face_term(std::vector<int> &octant_to_global, 
                           DoFInfo &dinfo1, DoFInfo &dinfo2,
                           CellInfo &info1, CellInfo &info2);

  std::vector<std::vector<int> > ordinates_in_octant;
  dealii::MeshWorker::IntegrationInfoBox<dim> info_box;
  Sweeper sweeper;
};

#endif  // AETHER_SN_TRANSPORT_H_