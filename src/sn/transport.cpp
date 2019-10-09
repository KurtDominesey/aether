#include "transport.hpp"
#include "quadrature.hpp"

#include <deal.II/dofs/dof_renumbering.h>

template <int dim, int qdim>
Transport<dim, qdim>::Transport(dealii::DoFHandler<dim> &dof_handler,
                                const dealii::Quadrature<qdim> &quadrature,
                                const std::vector<double> &cross_sections)
    : dof_handler(dof_handler),
      quadrature(quadrature),
      cross_sections(cross_sections) {
  const int num_ordinates = quadrature.get_points().size();
  ordinates.reserve(num_ordinates);
  for (int n = 0; n < num_ordinates; ++n) {
    ordinates.push_back(ordinate<dim>(quadrature.point(n)));
    std::cout << ordinates.back() << std::endl;
  }
  octant_directions.resize(std::pow(2, dim));
  renumberings.resize(
      octant_directions.size(),
      std::vector<dealii::types::global_dof_index>(dof_handler.n_dofs()));
  unrenumberings.resize(renumberings.size(), renumberings[0]);
  ordinates_in_octant.resize(octant_directions.size());
  if (dim == 1) {
    octant_directions[0] = dealii::Point<dim>(+1);
    octant_directions[1] = dealii::Point<dim>(-1);
    dealii::DoFRenumbering::compute_downstream(renumberings[0], 
                                               unrenumberings[0],
                                               dof_handler,
                                               octant_directions[0], false);
    renumberings[1] = renumberings[0];
    std::reverse(renumberings[1].begin(), renumberings[1].end());
    unrenumberings[1] = unrenumberings[0];
    std::reverse(unrenumberings[1].begin(), unrenumberings[1].end());
    // setup ordinates_in_octant
    for (int n = 0; n < num_ordinates; ++n) {
      const dealii::Point<qdim> &point = quadrature.point(n);
      int octant;
      if (point(0) > 0.5) octant = 0;
      else                octant = 1;
      ordinates_in_octant[octant].push_back(n);
    }
  } else if (dim == 2) {
    octant_directions[0] = dealii::Point<dim>(+1, +1);
    octant_directions[1] = dealii::Point<dim>(-1, +1);
    octant_directions[2] = dealii::Point<dim>(-1, -1);
    octant_directions[3] = dealii::Point<dim>(+1, -1);
    std::vector<int> opposites = {2, 3};
    for (int oct = 0; oct < 2; ++oct) {
      dealii::DoFRenumbering::compute_downstream(renumberings[oct], 
                                                 unrenumberings[oct],
                                                 dof_handler,
                                                 octant_directions[oct], false);
      int opposite = opposites[oct];
      renumberings[opposite] = renumberings[oct];
      std::reverse(renumberings[opposite].begin(), 
                   renumberings[opposite].end());
      unrenumberings[opposite] = unrenumberings[oct];
      std::reverse(unrenumberings[opposite].begin(), 
                   unrenumberings[opposite].end());
    }
    // setup ordinates_in_octant
    for (int n = 0; n < num_ordinates; ++n) {
      const dealii::Point<qdim> &point = quadrature.point(n);
      int octant;
      if (point(1) < 0.25)      octant = 0;
      else if (point(1) < 0.5)  octant = 1;
      else if (point(1) < 0.75) octant = 2;
      else                      octant = 4;
      Assert(point(0) > 0.5, dealii::ExcInvalidState());
      // 2D simulations allow only positive polar angles (symmetry)
      ordinates_in_octant[octant].push_back(n);
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
    std::vector<int> opposites = {6, 7, 4, 5};
    for (int oct = 0; oct < 4; ++oct) {
      dealii::DoFRenumbering::compute_downstream(renumberings[oct], 
                                                 unrenumberings[oct],
                                                 dof_handler,
                                                 octant_directions[oct], false);
      int opposite = opposites[oct];
      renumberings[opposite] = renumberings[oct];
      std::reverse(renumberings[opposite].begin(), 
                   renumberings[opposite].end());
      unrenumberings[opposite] = unrenumberings[oct];
      std::reverse(unrenumberings[opposite].begin(), 
                   unrenumberings[opposite].end());
    }
    // setup ordinates_in_octant
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
      ordinates_in_octant[octant].push_back(n);
    }
  }
}

template <int dim, int qdim>
void Transport<dim, qdim>::vmult(dealii::BlockVector<double> &dst,
                                 const dealii::BlockVector<double> &src) {
  int num_octants = octant_directions.size();
  for (int oct = 0; oct < num_octants; ++oct) {
    vmult_octant(oct, dst, src);
  }
}

template <int dim, int qdim>
void Transport<dim, qdim>::vmult_octant(int oct, 
                                        dealii::BlockVector<double> &dst,
                                        const dealii::BlockVector<double> &src) {
  dof_handler.renumber_dofs(renumberings[oct]);
  // setup finite elements
  dealii::FiniteElement<dim> fe = dof_handler.get_fe();
  dealii::QGauss<dim> quadrature(fe.degree+1);
  dealii::QGauss<dim-1> quadrature_face(fe.degree+1);
  const dealii::UpdateFlags update_flags = 
      dealii::update_values
      | dealii::update_gradients
      | dealii::update_quadrature_points
      | update_JxW_values;
  const dealii::UpdateFlags update_flags_face =
      dealii::update_values 
      | dealii::update_normal_vectors
      | dealii::update_quadrature_points
      | dealii::update_JxW_values;
  dealii::FEValues<dim> fe_values(fe, quadrature, update_flags);
  dealii::FEFaceValues<dim> fe_face_values(fe, quadrature_face,
                                           update_flags_face);
  dealii::FEFaceValues<dim> fe_face_values_neighbor(fe, quadrature_face,
                                                    update_flags_face);
  dealii::FESubfaceValues<dim> fe_subface_values(fe, quadrature_face, 
                                                 update_flags_face);
  std::vector<dealii::types::global_dof_index> dof_indices(fe.dofs_per_cell);
  // setup local storage
  dealii::FullMatrix<double> matrix(fe.dofs_per_cell);
  dealii::Vector<double> vector(fe.dofs_per_cell);
  typename dealii::DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();
  for (; cell != endc; cell++) {
    if (!cell->is_locally_owned()) continue;
    matrix = 0;
    vector = 0;
    fe_values.reinit(cell);
    // assemble_cell_terms
    cell->get_dof_indices(dof_indices);  // dof_indices are downwind
    for (int i = 0; i < dof_indices.size(); ++i)
      dof_indices[i] = unrenumberings[oct][dof_indices[i]];
    for (int f = 0; f < dealii::GeometryInfo<dim>::faces_per_cell; f++) {
      typename dealii::DoFHandler<dim>::face_iterator face = cell->face(f);
      if (face->at_boundary()) {
        // assemble boundary terms
      } else {
        Assert(cell->neighbor(f).state() == dealii::IteratorState::valid,
               dealii::ExcInvalidState());
        typename DofHandler<dim>::cell_iterator neighbor = cell->neighbor(f);
        if (face->has_children()) {
          const int f_neighbor = cell->neighbor_of_neighbor(f);
          for (int f_child = 0; f_child < face->number_of_children();
               ++f_child) {
            typename DoFHandler<dim>::cell_iterator neighbor_child = 
                cell->neighbor_child_on_subface(f, f_child);
            Assert(!neighbor_child->has_children(), dealii::ExcInvalidState());
            fe_subface_values.reinit(cell, f, f_child);
            fe_face_values_neighbor.reinit(neighbor_child, f_neighbor);
            // assemble face terms
          }
        } else { // face has no children
          const int f_neighbor = cell->neighbor_of_neighbor(f);
          fe_face_values.reinit(cell, f);
          fe_face_values_neighbor.reinit(neighbor, f_neighbor);
          // assemble face terms
        }
      }
    }
    // invert matrix
    // vmult vector
  }
  dof_handler.renumber_dofs(unrenumberings[oct]);
}

template <int dim, int qdim>
void Transport<dim, qdim>::integrate_cell_term(
    std::vector<int> &octant_to_global, DoFInfo &dinfo, CellInfo &info) {
  std::cout << "integrate cell\n";
  const dealii::FEValuesBase<dim> &fe_values = info.fe_values();
  const std::vector<double> &JxW = fe_values.get_JxW_values();
  int material = dinfo.cell->material_id();
  double cross_section = cross_sections[material];
  AssertDimension(octant_to_global.size(), dinfo.n_matrices());
  AssertDimension(octant_to_global.size(), dinfo.n_vectors());
  for (int n = 0; n < octant_to_global.size(); ++n) {
    dealii::FullMatrix<double> &local_matrix = dinfo.matrix(n).matrix;
    for (int q = 0; q < fe_values.n_quadrature_points; ++q) {
      for (int i = 0; i < fe_values.dofs_per_cell; ++i) {
        for (int j = 0; j < fe_values.dofs_per_cell; ++j) {
          double streaming = ordinates[octant_to_global[n]] *
                             fe_values.shape_grad(i, q) *
                             fe_values.shape_value(j, q);
          double collision = cross_section * fe_values.shape_value(i, q) *
                             fe_values.shape_value(j, q);
          local_matrix(i, j) += (-streaming + collision) * JxW[q];
        }
      }
    }
    std::cout << "ordinate " << n << " cell\n";
    local_matrix.print(std::cout);
  }
}

template <int dim, int qdim>
void Transport<dim, qdim>::integrate_boundary_term(
    std::vector<int> &octant_to_global, DoFInfo &dinfo, CellInfo &info) {
  std::cout << "integrate boundary\n";
  const dealii::FEValuesBase<dim> &fe_face_values = info.fe_values();
  const std::vector<double> &JxW = fe_face_values.get_JxW_values();
  const std::vector<dealii::Tensor<1, dim> > &normals =
      fe_face_values.get_normal_vectors();
  AssertDimension(octant_to_global.size(), dinfo.n_matrices());
  AssertDimension(octant_to_global.size(), dinfo.n_vectors());
  for (int n = 0; n < octant_to_global.size(); ++n) {
    dealii::FullMatrix<double> &local_matrix = dinfo.matrix(n).matrix;
    dealii::Vector<double> &local_vector = dinfo.vector(n).block(0);
    dealii::Vector<double> local_vector_copy = local_vector;
    for (int q = 0; q < fe_face_values.n_quadrature_points; ++q) {
      double ord_dot_normal = ordinates[octant_to_global[n]] * normals[q];
      if (ord_dot_normal > 0) {  // outflow
        for (int i = 0; i < fe_face_values.dofs_per_cell; ++i) {
          for (int j = 0; j < fe_face_values.dofs_per_cell; ++j) {
            local_matrix(i, j) += ord_dot_normal 
                                  * fe_face_values.shape_value(j, q)
                                  * fe_face_values.shape_value(i, q)
                                  * JxW[q];
          }
        }
      } else {  // inflow
        for (int i = 0; i < fe_face_values.dofs_per_cell; ++i) {
          for (int j = 0; j < fe_face_values.dofs_per_cell; ++j) {
            local_vector(i) += -ord_dot_normal
                               * 1.0 //local_vector_copy(j)
                               * fe_face_values.shape_value(j, q)
                               * fe_face_values.shape_value(i, q)
                               * JxW[q];
          }
        }
      }
    }
    // std::cout << "ordinate " << n << " boundary\n";
    // local_matrix.print(std::cout);
    // local_vector.print(std::cout);
  }
}

template <int dim, int qdim>
void Transport<dim, qdim>::integrate_face_term(
    std::vector<int> &octant_to_global, DoFInfo &dinfo1, DoFInfo &dinfo2,
    CellInfo &info1, CellInfo &info2) {
  std::cout << "integrate face\n";
  const dealii::FEValuesBase<dim> &fe_face_values = info1.fe_values();
  const int dofs_per_cell = fe_face_values.dofs_per_cell;
  const dealii::FEValuesBase<dim> &fe_face_values_neighbor = info2.fe_values();
  const int dofs_per_cell_neighbor = fe_face_values_neighbor.dofs_per_cell;
  const std::vector<double> &JxW = fe_face_values.get_JxW_values();
  const std::vector<dealii::Tensor<1, dim> > &normals =
      fe_face_values.get_normal_vectors();
  AssertDimension(octant_to_global.size(), dinfo1.n_matrices());
  AssertDimension(octant_to_global.size(), dinfo1.n_vectors());
  AssertDimension(octant_to_global.size(), dinfo2.n_matrices());
  AssertDimension(octant_to_global.size(), dinfo2.n_vectors());
  for (int n = 0; n < octant_to_global.size(); ++n) {
    dealii::FullMatrix<double> &outflow_matrix = dinfo1.matrix(n).matrix;
    dealii::Vector<double> &inflow_vector = dinfo1.vector(n).block(0);
    const dealii::Vector<double> &neighbor_vector = dinfo2.vector(n).block(0);
    for (int q = 0; q < fe_face_values.n_quadrature_points; ++q) {
      double ord_dot_normal = ordinates[octant_to_global[n]] * normals[q];
      if (ord_dot_normal > 0) {  // outflow
        for (int i = 0; i < dofs_per_cell; ++i) {
          for (int j = 0; j < dofs_per_cell; ++j) {
            outflow_matrix(i, j) += ord_dot_normal
                                  * fe_face_values.shape_value(j, q)
                                  * fe_face_values.shape_value(i, q)
                                  * JxW[q];
          }
        }
      } else {  // inflow
        for (int i = 0; i < dofs_per_cell; ++i) {
          for (int l = 0; l < dofs_per_cell_neighbor; ++l) {
            inflow_vector(i) += -ord_dot_normal
                                * 1.0 //neighbor_vector(l)
                                * fe_face_values_neighbor.shape_value(l, q)
                                * fe_face_values.shape_value(i, q)
                                * JxW[q];
          }
        }
      }
    }
    // std::cout << "ordinate " << n << " face\n";
    // outflow_matrix.print(std::cout);
    // inflow_vector.print(std::cout);
  }
}

template class Transport<1>;
template class Transport<2>;
template class Transport<3>;