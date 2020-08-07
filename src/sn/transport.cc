#include "transport.h"

namespace aether::sn {

template <int dim, int qdim>
Transport<dim, qdim>::Transport(
    const dealii::DoFHandler<dim> &dof_handler,
    const QAngle<dim, qdim> &quadrature)
    : dof_handler(dof_handler),
      quadrature(quadrature) {
  const int num_ordinates = quadrature.get_points().size();
  ordinates.reserve(num_ordinates);
  for (int n = 0; n < num_ordinates; ++n) {
    ordinates.push_back(ordinate<dim>(quadrature.point(n)));
  }
  const int num_octants = std::pow(2, dim);
  octant_directions.resize(num_octants);
  octants_to_global.resize(num_octants);
  // populate cell vector
  cells.reserve(dof_handler.get_triangulation().n_active_cells());
  for (auto cell = dof_handler.begin_active(); cell != dof_handler.end(); 
       ++cell)
    cells.push_back(cell);
  // populate sweep orders with unordered (z-ordered) indices
  sweep_orders.resize(1);
  sweep_orders[0].resize(cells.size());
  std::iota(sweep_orders[0].begin(), sweep_orders[0].end(), 0);
  sweep_orders.resize(num_ordinates, sweep_orders[0]);
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
      else                      octant = 3;
      // Assert(point(0) > 0.5, 
      //        dealii::ExcMessage("2D simulations allow only positive polar "
      //                           "angles. Impose polar symmetry on "
      //                           "quadrature."));
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
  for (int n = 0; n < num_ordinates; ++n) {
    const dealii::DoFRenumbering::CompareDownstream<ActiveCell, dim>
        comparator(ordinates[n]);
    auto compare = [&comparator, this](const int &a, const int &b) -> bool {
      return comparator(this->cells[a], this->cells[b]);
    };
    std::sort(sweep_orders[n].begin(), sweep_orders[n].end(), compare);
  }
  assemble_cell_matrices();
}

template <int dim, int qdim>
void Transport<dim, qdim>::assemble_cell_matrices() {
  Assert(cell_matrices.empty(), dealii::ExcInvalidState());
  const dealii::FiniteElement<dim> &fe = dof_handler.get_fe();
  const dealii::Triangulation<dim> &mesh = dof_handler.get_triangulation();
  // setup finite elements
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
  for (auto cell = mesh.begin_active(); cell != mesh.end(); ++cell) {
    if (!cell->is_locally_owned())
      continue;
    fe_values.reinit(cell);
    const std::vector<double> &JxW = fe_values.get_JxW_values();
    cell_matrices.emplace_back(fe_values.dofs_per_cell,
                               dealii::GeometryInfo<dim>::faces_per_cell,
                               quadrature_face.size());
    auto &matrices = cell_matrices.back();
    for (int q = 0; q < fe_values.n_quadrature_points; ++q) {
      for (int i = 0; i < fe_values.dofs_per_cell; ++i) {
        for (int j = 0; j < fe_values.dofs_per_cell; ++j) {
          matrices.mass[i][j] += fe_values.shape_value(i, q) *
                                 fe_values.shape_value(j, q) *
                                 JxW[q];
          matrices.grad[i][j] += fe_values.shape_grad(i, q) *
                                 fe_values.shape_value(j, q) *
                                 JxW[q];
        }
      }
    }
    for (int f = 0; f < dealii::GeometryInfo<dim>::faces_per_cell; ++f) {
      auto face = cell->face(f);
      fe_face_values.reinit(cell, f);
      const std::vector<dealii::Tensor<1, dim>> &normals =
          fe_face_values.get_normal_vectors();
      const std::vector<double> &JxW_face = fe_face_values.get_JxW_values();
      for (int q = 0; q < fe_face_values.n_quadrature_points; ++q) {
        matrices.normals[f][q] = normals[q];
        for (int i = 0; i < fe_face_values.dofs_per_cell; ++i)
          for (int j = 0; j < fe_face_values.dofs_per_cell; ++j)
            matrices.outflow[f][i][j] += normals[q] *
                                         fe_face_values.shape_value(i, q) *
                                         fe_face_values.shape_value(j, q) *
                                         JxW_face[q];
      }
      if (face->at_boundary())
        continue;
      const auto neighbor = cell->neighbor(f);
      const int f_neighbor = cell->neighbor_of_neighbor(f);
      if (!face->has_children()) {
        fe_face_values_neighbor.reinit(neighbor, f_neighbor);
        matrices.inflow[f].emplace_back(fe_face_values.dofs_per_cell,
                                        fe_face_values_neighbor.dofs_per_cell);
        for (int q = 0; q < fe_face_values.n_quadrature_points; ++q)
          for (int i = 0; i < fe_face_values.dofs_per_cell; ++i)
            for (int j = 0; j < fe_face_values_neighbor.dofs_per_cell; ++j)
              matrices.inflow[f][0][i][j] +=
                  normals[q] *
                  fe_face_values.shape_value(i, q) *
                  fe_face_values_neighbor.shape_value(j, q) *
                  JxW_face[q];
      } else {
        for (int f_sub = 0; f_sub < face->number_of_children(); ++f_sub) {
          const auto neighbor_child = cell->neighbor_child_on_subface(f, f_sub);
          fe_subface_values.reinit(cell, f, f_sub);
          fe_face_values_neighbor.reinit(neighbor_child, f_neighbor);
          const std::vector<dealii::Tensor<1, dim>> &subnormals =
              fe_subface_values.get_normal_vectors();
          matrices.inflow[f].emplace_back(fe_subface_values.dofs_per_cell,
                                          fe_face_values_neighbor.dofs_per_cell);
          for (int q = 0; q < fe_subface_values.n_quadrature_points; ++q)
            for (int i = 0; i < fe_subface_values.dofs_per_cell; ++i)
              for (int j = 0; j < fe_face_values_neighbor.dofs_per_cell; ++j)
                matrices.inflow[f][f_sub][i][j] +=
                    subnormals[q] *
                    fe_subface_values.shape_value(i, q) *
                    fe_face_values_neighbor.shape_value(j, q) *
                    JxW_face[q];
        }
      }
    }
  }
}

template <int dim, int qdim>
void Transport<dim, qdim>::vmult(dealii::Vector<double> &dst,
                                 const dealii::Vector<double> &src,
                                 const std::vector<double> &cross_sections,
                                 const std::vector<dealii::BlockVector<double>>
                                     &boundary_conditions) const {
  dealii::BlockVector<double> dst_b(quadrature.size(), dof_handler.n_dofs());
  dealii::BlockVector<double> src_b(quadrature.size(), dof_handler.n_dofs());
  dst_b = dst;
  src_b = src;
  vmult(dst_b, src_b, cross_sections, boundary_conditions);
  dst = dst_b;
}

template <int dim, int qdim>
void Transport<dim, qdim>::vmult(dealii::BlockVector<double> &dst,
                                 const dealii::BlockVector<double> &src,
                                 const std::vector<double> &cross_sections,
                                 const std::vector<dealii::BlockVector<double>>
                                     &boundary_conditions) const {
  const std::vector<dealii::types::boundary_id> &boundaries =
      dof_handler.get_triangulation().get_boundary_ids();
  int num_natural_boundaries = boundaries.size();
  for (const dealii::types::boundary_id &boundary : boundaries)
    if (boundary == types::reflecting_boundary_id)
      num_natural_boundaries--;
  for (const dealii::types::boundary_id &boundary : boundaries)
    if (boundary != types::reflecting_boundary_id)
      AssertIndexRange(boundary, num_natural_boundaries);
  AssertDimension(num_natural_boundaries, boundary_conditions.size());
  int num_octants = octant_directions.size();
  for (int oct = 0; oct < num_octants; ++oct) {
    vmult_octant(oct, dst, src, cross_sections, boundary_conditions);
  }
}

template <int dim, int qdim>
void Transport<dim, qdim>::vmult_octant(
    int oct, dealii::BlockVector<double> &dst,
    const dealii::BlockVector<double> &src,
    const std::vector<double> &cross_sections,
    const std::vector<dealii::BlockVector<double>> &boundary_conditions) const {
  const std::vector<int> &octant_to_global = octants_to_global[oct];
  const dealii::FiniteElement<dim> &fe = dof_handler.get_fe();
  std::vector<dealii::types::global_dof_index> dof_indices(fe.dofs_per_cell);
  std::vector<dealii::types::global_dof_index> dof_indices_neighbor(
      fe.dofs_per_cell);
  // assert each ordinate belongs to this octant
  double norm_a = octant_directions[oct].norm();
  for (int n_oct = 0; n_oct < octant_to_global.size(); ++n_oct) {
    const Ordinate &ordinate = ordinates[octant_to_global[n_oct]];
    double cos_angle =
        (octant_directions[oct] * ordinate) / (norm_a * ordinate.norm());
    double angle = std::abs(std::acos(cos_angle));
    Assert(angle <= double{dealii::numbers::PI_4},
           dealii::ExcMessage("Ordinate not in octant"));
    Assert(ordinate.norm() - 1.0 < 1e-12, 
           dealii::ExcMessage("Ordinate not of unit magnitude"));
  }
  // setup local storage
  dealii::FullMatrix<double> matrix(fe.dofs_per_cell);
  dealii::Vector<double> rhs_cell(fe.dofs_per_cell);
  dealii::Vector<double> src_cell(fe.dofs_per_cell);
  dealii::Vector<double> dst_cell(fe.dofs_per_cell);
  dealii::Vector<double> dst_boundary(fe.dofs_per_cell);
  for (int n_oct = 0; n_oct < octant_to_global.size(); ++n_oct) {
    const int n = octant_to_global[n_oct];
    const Ordinate &ordinate = ordinates[n];
    for (int c : sweep_orders[n]) {
      const ActiveCell &cell = cells[c];
      if (!cell->is_locally_owned()) 
        continue;
      cell->get_dof_indices(dof_indices);
      int material = cell->material_id();
      double cross_section = cross_sections[material];
      const auto &matrices = cell_matrices[c];
      // assemble volume source
      for (int i = 0; i < dof_indices.size(); ++i)
        src_cell[i] = src.block(n)[dof_indices[i]];
      matrices.mass.vmult(rhs_cell, src_cell);
      // assemble volume integrals
      for (int i = 0; i < dof_indices.size(); ++i)
        for (int j = 0; j < dof_indices.size(); ++j)
          matrix[i][j] = -(ordinate * matrices.grad[i][j])
                         + cross_section * matrices.mass[i][j];
      for (int f = 0; f < dealii::GeometryInfo<dim>::faces_per_cell; ++f) {
        // assemble face integrals
        double ord_dot_normal = 0;
        for (int q = 0; q < matrices.normals.size(1); ++q) {
          double ord_dot_normal_q = ordinate * matrices.normals[f][q];
          if (ord_dot_normal_q == 0)
            continue;
          if (ord_dot_normal == 0)
            ord_dot_normal = ord_dot_normal_q;
          else
            Assert(ord_dot_normal_q * ord_dot_normal > 0, 
                   dealii::ExcMessage("Face is re-entrant"));
        }
        // outflow
        if (ord_dot_normal > 0) {
          for (int i = 0; i < dof_indices.size(); ++i)
            for (int j = 0; j < dof_indices.size(); ++j)
              matrix[i][j] += ordinate * matrices.outflow[f][i][j];
        // inflow
        } else if (ord_dot_normal < 0) {
          const Face face = cell->face(f);
          if (!face->at_boundary()) {
            // inflow from neighbor
            const Cell neighbor = cell->neighbor(f);
            const int f_neighbor = cell->neighbor_of_neighbor(f);
            Assert(!face->has_children(), dealii::ExcNotImplemented());
            if (!face->has_children()) {
              // one neighbor
              neighbor->get_dof_indices(dof_indices_neighbor);
              for (int i = 0; i < dof_indices.size(); ++i)
                for (int j = 0; j < dof_indices_neighbor.size(); ++j)
                  rhs_cell[i] -= ordinate * matrices.inflow[f][0][i][j] *
                                 dst.block(n)[dof_indices_neighbor[j]];
            } else {
              // multiple neighbors
              for (int f_sub = 0; f_sub < face->number_of_children(); ++f_sub) {
                const Cell &neighbor_child =
                    cell->neighbor_child_on_subface(f, f_sub);
                neighbor_child->get_dof_indices(dof_indices_neighbor);
                for (int i = 0; i < dof_indices.size(); ++i)
                  for (int j = 0; j < dof_indices_neighbor.size(); ++j)
                    rhs_cell[i] -= ordinate * matrices.inflow[f][f_sub][i][j] *
                                   dst.block(n)[dof_indices_neighbor[j]];
              }
            }
          } else {  // face->at_boundary()
            // inflow from boundary
            if (face->boundary_id() == types::reflecting_boundary_id) {
             int n_refl = quadrature.reflected_index(n, matrices.normals[f][0]);
              for (int j = 0; j < dof_indices.size(); ++j)
                dst_boundary[j] = dst.block(n_refl)[dof_indices[j]];
            } else {
              dst_boundary = boundary_conditions[face->boundary_id()].block(n);
            }
            for (int i = 0; i < dof_indices.size(); ++i)
              for (int j = 0; j < dof_indices.size(); ++j)
                rhs_cell[i] -=
                    ordinate * matrices.outflow[f][i][j] * dst_boundary[j];
          }
        }
      }
      matrix.gauss_jordan();  // directly invert
      matrix.vmult(dst_cell, rhs_cell);
      for (int i = 0; i < dof_indices.size(); ++i)
        dst.block(n)[dof_indices[i]] = dst_cell[i];
    }
  }
}

template <int dim, int qdim>
int Transport<dim, qdim>::n_block_rows() const {
  return ordinates.size();
}

template <int dim, int qdim>
int Transport<dim, qdim>::n_block_cols() const {
  return ordinates.size();
}

template <int dim, int qdim>
dealii::BlockIndices Transport<dim, qdim>::get_block_indices() const {
  return dealii::BlockIndices(quadrature.size(), dof_handler.n_dofs());
}

template class Transport<1>;
template class Transport<2>;
template class Transport<3>;

template struct CellMatrices<1>;
template struct CellMatrices<2>;
template struct CellMatrices<3>;

}  // namespace aether::sn