#include "transport.hpp"

template <int dim, int qdim>
Transport<dim, qdim>::Transport(
    dealii::DoFHandler<dim> &dof_handler,
    const dealii::Quadrature<qdim> &quadrature,
    const std::vector<double> &cross_sections,
    const std::vector<dealii::BlockVector<double>> &boundary_conditions)
    : dof_handler(dof_handler),
      quadrature(quadrature),
      cross_sections(cross_sections),
      boundary_conditions(boundary_conditions) {
  const std::vector<dealii::types::boundary_id> &boundaries =
      dof_handler.get_triangulation().get_boundary_ids();
  for (const dealii::types::boundary_id &boundary : boundaries)
    AssertIndexRange(boundary, boundaries.size());
  AssertDimension(boundaries.size(), boundary_conditions.size());
  const int num_ordinates = quadrature.get_points().size();
  ordinates.reserve(num_ordinates);
  for (int n = 0; n < num_ordinates; ++n) {
    ordinates.push_back(ordinate<dim>(quadrature.point(n)));
  }
  const int num_octants = std::pow(2, dim);
  octant_directions.resize(num_octants);
  octants_to_global.resize(num_octants);
  // populate downstream cell vectors with unordered (z-ordered) cells
  cells_downstream.resize(1);
  cells_downstream[0].reserve(dof_handler.get_triangulation().n_active_cells());
  for (ActiveCell cell = dof_handler.begin_active(); cell != dof_handler.end();
       ++cell)
    cells_downstream[0].push_back(cell);
  cells_downstream.resize(num_octants, cells_downstream[0]);
  if (dim == 1) {
    octant_directions[0] = dealii::Point<dim>(+1);
    octant_directions[1] = dealii::Point<dim>(-1);
    // setup ordinates_in_octant
    for (int n = 0; n < num_ordinates; ++n) {
      const dealii::Point<qdim> &point = quadrature.point(n);
      int octant;
      if (point(0) > 0.5) octant = 0;
      else                octant = 1;
      octants_to_global[octant].push_back(n);
    }
  } else if (dim == 2) {
    octant_directions[0] = dealii::Point<dim>(+1, +1);
    octant_directions[1] = dealii::Point<dim>(-1, +1);
    octant_directions[2] = dealii::Point<dim>(-1, -1);
    octant_directions[3] = dealii::Point<dim>(+1, -1);
    // std::vector<int> opposites = {2, 3};
    // setup ordinates_in_octants
    for (int n = 0; n < num_ordinates; ++n) {
      const dealii::Point<qdim> &point = quadrature.point(n);
      int octant;
      if (point(1) < 0.25)      octant = 0;
      else if (point(1) < 0.5)  octant = 1;
      else if (point(1) < 0.75) octant = 2;
      else                      octant = 4;
      Assert(point(0) > 0.5, dealii::ExcInvalidState());
      // 2D simulations allow only positive polar angles (symmetry)
      octants_to_global[octant].push_back(n);
    }
  } else if (dim == 3) {
    octant_directions[0] = dealii::Point<dim>(+1, +1, +1);
    octant_directions[1] = dealii::Point<dim>(-1, +1, +1);
    octant_directions[2] = dealii::Point<dim>(-1, -1, +1);
    octant_directions[3] = dealii::Point<dim>(+1, -1, +1);
    octant_directions[4] = dealii::Point<dim>(+1, +1, -1);
    octant_directions[5] = dealii::Point<dim>(-1, +1, -1);
    octant_directions[6] = dealii::Point<dim>(-1, -1, -1);
    octant_directions[7] = dealii::Point<dim>(+1, -1, -1);
    // std::vector<int> opposites = {6, 7, 4, 5};
    // setup ordinates_in_octants
    for (int n = 0; n < num_ordinates; ++n) {
      const dealii::Point<qdim> &point = quadrature.point(n);
      int octant;
      if (point(0) > 0.5) {
        if (point(1) < 0.25)      octant = 0;
        else if (point(1) < 0.5)  octant = 1;
        else if (point(1) < 0.75) octant = 2;
        else                      octant = 3;
      } else {
        if (point(1) < 0.25)      octant = 4;
        else if (point(1) < 0.5)  octant = 5;
        else if (point(1) < 0.75) octant = 6;
        else                      octant = 7;
      }
      octants_to_global[octant].push_back(n);
    }
  }
  for (int octant = 0; octant < octant_directions.size(); ++octant) {
    const dealii::DoFRenumbering::CompareDownstream<ActiveCell, dim>
        comparator(octant_directions[octant]);
    std::sort(cells_downstream[octant].begin(), cells_downstream[octant].end(),
              comparator);
  }
}

template <int dim, int qdim>
void Transport<dim, qdim>::vmult(dealii::BlockVector<double> &dst,
                                 const dealii::BlockVector<double> &src,
                                 const bool homogeneous) const {
  dst = 0;
  int num_octants = octant_directions.size();
  for (int oct = 0; oct < num_octants; ++oct) {
    vmult_octant(oct, dst, src, homogeneous);
  }
}

template <int dim, int qdim>
void Transport<dim, qdim>::vmult_octant(int oct, 
                                        dealii::BlockVector<double> &dst,
                                        const dealii::BlockVector<double> &src,
                                        const bool homogeneous) 
                                        const {
  const std::vector<int> &octant_to_global = octants_to_global[oct];
  // setup finite elements
  const dealii::FiniteElement<dim> &fe = dof_handler.get_fe();
  dealii::QGauss<dim> quadrature_fe(fe.degree+1);
  dealii::QGauss<dim-1> quadrature_face(fe.degree+1);
  const dealii::UpdateFlags update_flags = 
      dealii::update_values
      | dealii::update_gradients
      | dealii::update_quadrature_points
      | dealii::update_JxW_values;
  const dealii::UpdateFlags update_flags_face =
      dealii::update_values 
      | dealii::update_normal_vectors
      | dealii::update_quadrature_points
      | dealii::update_JxW_values;
  dealii::FEValues<dim> fe_values(fe, quadrature_fe, update_flags);
  dealii::FEFaceValues<dim> fe_face_values(fe, quadrature_face,
                                           update_flags_face);
  dealii::FEFaceValues<dim> fe_face_values_neighbor(fe, quadrature_face,
                                                    update_flags_face);
  dealii::FESubfaceValues<dim> fe_subface_values(fe, quadrature_face, 
                                                 update_flags_face);
  std::vector<dealii::types::global_dof_index> dof_indices(fe.dofs_per_cell);
  std::vector<dealii::types::global_dof_index> dof_indices_neighbor(
      fe.dofs_per_cell);
  // get ordinates
  const int num_ords = octant_to_global.size();  // only ords in oct
  std::vector<Ordinate> ordinates_in_octant;
  // ordinates_in_octant.reserve(num_ords);
  for (int n = 0; n < num_ords; ++n)
    ordinates_in_octant.push_back(ordinates[octant_to_global[n]]);
  // assert each ordinate belongs to this octant
  double norm_a = octant_directions[oct].norm();
  for (int n = 0; n < ordinates_in_octant.size(); ++n) {
    Ordinate &ordinate = ordinates_in_octant[n];
    double cos_angle =
        (octant_directions[oct] * ordinate) / (norm_a * ordinate.norm());
    Assert(std::acos(cos_angle) <= dealii::numbers::PI_4,
           dealii::ExcMessage(std::to_string(cos_angle)));
  }
  // setup local storage
  std::vector<dealii::FullMatrix<double>> matrices(
      num_ords, dealii::FullMatrix<double>(fe.dofs_per_cell));
  dealii::BlockVector<double> rhs_cell(num_ords, fe.dofs_per_cell);
  dealii::BlockVector<double> src_cell(num_ords, fe.dofs_per_cell);
  dealii::BlockVector<double> dst_cell(num_ords, fe.dofs_per_cell);
  dealii::BlockVector<double> dst_neighbor(num_ords, fe.dofs_per_cell);
  std::vector<dealii::BlockVector<double>> boundary_conditions_incident(
      boundary_conditions.size(), 
      dealii::BlockVector<double>(num_ords, fe.dofs_per_cell));
  for (int b = 0; b < boundary_conditions.size(); ++b)
    for (int n = 0; n < num_ords; ++n)
      boundary_conditions_incident[b].block(n) =
          boundary_conditions[b].block(octant_to_global[n]);
  for (const ActiveCell &cell : cells_downstream[oct]) {
    if (!cell->is_locally_owned()) 
      continue;
    cell->get_dof_indices(dof_indices);
    for (int n = 0; n < num_ords; ++n) {
      matrices[n] = 0;
      src_cell.block(n) = 0;
      for (int i = 0; i < fe.dofs_per_cell; ++i)
        rhs_cell.block(n)[i] = src.block(octant_to_global[n])[dof_indices[i]];
    }
    fe_values.reinit(cell);
    int material = cell->material_id();
    double cross_section = cross_sections[material];
    integrate_cell_term(ordinates_in_octant, fe_values, rhs_cell, cross_section,
                        matrices, src_cell);
    for (int f = 0; f < dealii::GeometryInfo<dim>::faces_per_cell; f++) {
      const Face &face = cell->face(f);
      if (face->at_boundary()) {
        fe_face_values.reinit(cell, f);
        dealii::BlockVector<double> &dst_boundary =
            boundary_conditions_incident[face->boundary_id()];
        integrate_boundary_term(ordinates_in_octant, fe_face_values, 
                                dst_boundary, matrices, src_cell, homogeneous);
      } else {
        Assert(cell->neighbor(f).state() == dealii::IteratorState::valid,
               dealii::ExcInvalidState());
        const Cell &neighbor = cell->neighbor(f);
        if (face->has_children()) {
          Assert(false, dealii::ExcNotImplemented());
          const int f_neighbor = cell->neighbor_of_neighbor(f);
          for (int f_sub = 0; f_sub < face->number_of_children(); ++f_sub) {
            const Cell &neighbor_child = 
                cell->neighbor_child_on_subface(f, f_sub);
            neighbor_child->get_dof_indices(dof_indices_neighbor);
            for (int n = 0; n < num_ords; ++n)
              for (int i = 0; i < fe.dofs_per_cell; ++i)
                dst_neighbor.block(n)[i] =
                    dst.block(octant_to_global[n])[dof_indices_neighbor[i]];
            Assert(!neighbor_child->has_children(), dealii::ExcInvalidState());
            fe_subface_values.reinit(cell, f, f_sub);
            fe_face_values_neighbor.reinit(neighbor_child, f_neighbor);
            integrate_face_term(ordinates_in_octant, fe_subface_values,
                                fe_face_values_neighbor, dst_neighbor, matrices,
                                src_cell);
          }
        } else { // !face->has_children()
          neighbor->get_dof_indices(dof_indices_neighbor);
          for (int n = 0; n < num_ords; ++n)
            for (int i = 0; i < fe.dofs_per_cell; ++i)
              dst_neighbor.block(n)[i] =
                  dst.block(octant_to_global[n])[dof_indices_neighbor[i]];
          const int f_neighbor = cell->neighbor_of_neighbor(f);
          fe_face_values.reinit(cell, f);
          fe_face_values_neighbor.reinit(neighbor, f_neighbor);
          integrate_face_term(ordinates_in_octant, fe_face_values,
                              fe_face_values_neighbor, dst_neighbor, matrices,
                              src_cell);
        }
      }
    }
    for (int n = 0; n < num_ords; ++n) {
      dealii::FullMatrix<double> &matrix = matrices[n];
      matrix.gauss_jordan();  // directly invert
      matrix.vmult(dst_cell.block(n), src_cell.block(n));
      dst.block(octant_to_global[n]).add(dof_indices, dst_cell.block(n));
    }
  }
}

template <int dim, int qdim>
void Transport<dim, qdim>::integrate_cell_term(
    const std::vector<Ordinate> &ordinates_in_sweep,
    const dealii::FEValues<dim> &fe_values,
    const dealii::BlockVector<double> &rhs_cell, double cross_section,
    std::vector<dealii::FullMatrix<double>> &matrices,
    dealii::BlockVector<double> &src_cell) const {
  const std::vector<double> &JxW = fe_values.get_JxW_values();
  for (int n = 0; n < ordinates_in_sweep.size(); ++n) {
    const Ordinate &ordinate = ordinates_in_sweep[n];
    dealii::FullMatrix<double> &matrix = matrices[n];
    for (int q = 0; q < fe_values.n_quadrature_points; ++q) {
      for (int i = 0; i < fe_values.dofs_per_cell; ++i) {
        for (int j = 0; j < fe_values.dofs_per_cell; ++j) {
          double streaming = ordinate * fe_values.shape_grad(i, q) *
                             fe_values.shape_value(j, q);
          double collision = cross_section * fe_values.shape_value(i, q) *
                             fe_values.shape_value(j, q);
          matrix(i, j) += (-streaming + collision) * JxW[q];
          src_cell.block(n)(i) += rhs_cell.block(n)[i] 
                                  * fe_values.shape_value(i, q)
                                  * fe_values.shape_value(j, q) 
                                  * JxW[q];
        }
      }
    }
  }
}

template <int dim, int qdim>
void Transport<dim, qdim>::integrate_boundary_term(
    const std::vector<Ordinate> &ordinates_in_sweep,
    const dealii::FEFaceValues<dim> &fe_face_values,
    const dealii::BlockVector<double> &dst_boundary,
    std::vector<dealii::FullMatrix<double>> &matrices,
    dealii::BlockVector<double> &src_cell,
    const bool homogeneous) const {
  const std::vector<double> &JxW = fe_face_values.get_JxW_values();
  const std::vector<dealii::Tensor<1, dim>> &normals =
      fe_face_values.get_normal_vectors();
  for (int n = 0; n < ordinates_in_sweep.size(); ++n) {
    const Ordinate &ordinate = ordinates_in_sweep[n];
    dealii::FullMatrix<double> &matrix = matrices[n];
    for (int q = 0; q < fe_face_values.n_quadrature_points; ++q) {
      double ord_dot_normal = ordinate * normals[q];
      if (ord_dot_normal > 0) {  // outflow
        for (int i = 0; i < fe_face_values.dofs_per_cell; ++i) {
          for (int j = 0; j < fe_face_values.dofs_per_cell; ++j) {
            matrix(i, j) += ord_dot_normal 
                            * fe_face_values.shape_value(j, q)
                            * fe_face_values.shape_value(i, q)
                            * JxW[q];
          }
        }
      } else if (!homogeneous) {  // inflow
        for (int i = 0; i < fe_face_values.dofs_per_cell; ++i) {
          for (int j = 0; j < fe_face_values.dofs_per_cell; ++j) {
            src_cell.block(n)(i) += -ord_dot_normal
                                    * dst_boundary.block(n)(j)
                                    * fe_face_values.shape_value(j, q)
                                    * fe_face_values.shape_value(i, q)
                                    * JxW[q];
          }
        }
      }
    }
  }
}

template <int dim, int qdim>
void Transport<dim, qdim>::integrate_face_term(
    const std::vector<Ordinate> &ordinates_in_sweep,
    const dealii::FEFaceValuesBase<dim> &fe_face_values,
    const dealii::FEFaceValuesBase<dim> &fe_face_values_neighbor,
    const dealii::BlockVector<double> &dst_neighbor,
    std::vector<dealii::FullMatrix<double>> &matrices,
    dealii::BlockVector<double> &src_cell) const {
  const std::vector<double> &JxW = fe_face_values.get_JxW_values();
  const std::vector<dealii::Tensor<1, dim> > &normals =
      fe_face_values.get_normal_vectors();
  for (int n = 0; n < ordinates_in_sweep.size(); ++n) {
    const Ordinate &ordinate = ordinates_in_sweep[n];
    dealii::FullMatrix<double> &matrix = matrices[n];
    for (int q = 0; q < fe_face_values.n_quadrature_points; ++q) {
      double ord_dot_normal = ordinate * normals[q];
      if (ord_dot_normal > 0) {  // outflow
        for (int k = 0; k < fe_face_values_neighbor.dofs_per_cell; ++k)
          Assert(dst_neighbor.block(n)(k) == 0.0, dealii::ExcInvalidState());
        for (int i = 0; i < fe_face_values.dofs_per_cell; ++i) {
          for (int j = 0; j < fe_face_values.dofs_per_cell; ++j) {
            matrix(i, j) += ord_dot_normal
                             * fe_face_values.shape_value(j, q)
                             * fe_face_values.shape_value(i, q)
                             * JxW[q];
          }
        }
      } else {  // inflow
        for (int i = 0; i < fe_face_values.dofs_per_cell; ++i) {
          for (int k = 0; k < fe_face_values_neighbor.dofs_per_cell; ++k) {
            src_cell.block(n)(i) += -ord_dot_normal
                                    * dst_neighbor.block(n)(k)
                                    * fe_face_values_neighbor.shape_value(k, q)
                                    * fe_face_values.shape_value(i, q)
                                    * JxW[q];
          }
        }
      }
    }
  }
}

template class Transport<1>;
template class Transport<2>;
template class Transport<3>;