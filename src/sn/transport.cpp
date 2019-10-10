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
  octants_to_global.resize(octant_directions.size());
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
      octants_to_global[octant].push_back(n);
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
    for (dealii::types::global_dof_index i = 0; i < renumberings[octant].size();
         ++i) {
      // std::cout << i << " " 
      //           << renumberings[octant][i] << " "
      //           << unrenumberings[octant][renumberings[octant][i]] << std::endl;
      AssertDimension(unrenumberings[octant][renumberings[octant][i]], i);
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
  std::vector<dealii::types::global_dof_index> &renumbering = renumberings[oct];
  std::vector<dealii::types::global_dof_index> &unrenumbering = 
      unrenumberings[oct];
  std::vector<int> &octant_to_global = octants_to_global[oct];
  dof_handler.renumber_dofs(renumbering);
  // setup finite elements
  const dealii::FiniteElement<dim> &fe = dof_handler.get_fe();
  dealii::QGauss<dim> quadrature(fe.degree+1);
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
  dealii::FEValues<dim> fe_values(fe, quadrature, update_flags);
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
  ordinates_in_octant.reserve(num_ords);
  for (int n = 0; n < num_ords; ++n)
    ordinates_in_octant.push_back(ordinates[octant_to_global[n]]);
  // setup local storage
  std::vector<dealii::FullMatrix<double>> matrices(
      num_ords, dealii::FullMatrix<double>(fe.dofs_per_cell));
  dealii::BlockVector<double> src_cell(num_ords, fe.dofs_per_cell);
  dealii::BlockVector<double> dst_cell(num_ords, fe.dofs_per_cell);
  dealii::BlockVector<double> dst_neighbor(num_ords, fe.dofs_per_cell);
  using ActiveCell = typename dealii::DoFHandler<dim>::active_cell_iterator;
  using Cell = typename dealii::DoFHandler<dim>::cell_iterator;
  using Face = typename dealii::DoFHandler<dim>::face_iterator;
  for (ActiveCell cell = dof_handler.begin_active(); cell != dof_handler.end();
       cell++) {
    if (!cell->is_locally_owned()) continue;
    cell->get_dof_indices(dof_indices);  // dof_indices are downwind
    for (int i = 0; i < dof_indices.size(); ++i)
      dof_indices[i] = unrenumbering[dof_indices[i]];
    for (int n = 0; n < num_ords; ++n) {
      matrices[n] = 0;
      for (dealii::types::global_dof_index dof_index : dof_indices)
        src_cell.block(n) = src.block(octant_to_global[n])[dof_index];
    }
    fe_values.reinit(cell);
    int material = cell->material_id();
    double cross_section = cross_sections[material];
    integrate_cell_term(ordinates_in_octant, fe_values, cross_section,
                        matrices);
    for (int f = 0; f < dealii::GeometryInfo<dim>::faces_per_cell; f++) {
      Face face = cell->face(f);
      if (face->at_boundary()) {
        fe_face_values.reinit(cell, f);
        integrate_boundary_term(ordinates_in_octant, fe_face_values, matrices,
                                src_cell);
      } else {
        Assert(cell->neighbor(f).state() == dealii::IteratorState::valid,
               dealii::ExcInvalidState());
        Cell neighbor = cell->neighbor(f);
        if (face->has_children()) {
          const int f_neighbor = cell->neighbor_of_neighbor(f);
          for (int f_sub = 0; f_sub < face->number_of_children(); ++f_sub) {
            Cell neighbor_child = cell->neighbor_child_on_subface(f, f_sub);
            neighbor_child->get_dof_indices(dof_indices_neighbor);  // downwind
            for (int i = 0; i < dof_indices_neighbor.size(); ++i)
              dof_indices_neighbor[i] = unrenumbering[dof_indices_neighbor[i]];
            for (int n = 0; n < num_ords; ++n)
              for (dealii::types::global_dof_index dof_index :
                   dof_indices_neighbor)
                dst_neighbor.block(n) =
                    dst.block(octant_to_global[n])[dof_index];
            Assert(!neighbor_child->has_children(), dealii::ExcInvalidState());
            fe_subface_values.reinit(cell, f, f_sub);
            fe_face_values_neighbor.reinit(neighbor_child, f_neighbor);
            integrate_face_term(ordinates_in_octant, fe_subface_values,
                                fe_face_values_neighbor, dst_neighbor, matrices,
                                src_cell);
          }
        } else { // !face->has_children()
          neighbor->get_dof_indices(dof_indices_neighbor);  // downwind
          for (int i = 0; i < dof_indices_neighbor.size(); ++i)
            dof_indices_neighbor[i] = unrenumbering[dof_indices_neighbor[i]];
          for (int n = 0; n < num_ords; ++n)
            for (dealii::types::global_dof_index dof_index :
                 dof_indices_neighbor)
              dst_neighbor.block(n) = dst.block(octant_to_global[n])[dof_index];
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
      // matrix.print(std::cout);
      // std::cout << std::endl;
      matrix.gauss_jordan();  // directly invert
      matrix.vmult(dst_cell.block(n), src_cell.block(n));
      dst.block(octant_to_global[n]).add(dof_indices, dst_cell.block(n));
      // dst_cell.print(std::cout);
      // std::cout << std::endl;
    }
  }
  dof_handler.renumber_dofs(unrenumbering);
}

template <int dim, int qdim>
void Transport<dim, qdim>::integrate_cell_term(
    const std::vector<Ordinate> &ordinates_in_sweep,
    const dealii::FEValues<dim> &fe_values, double cross_section,
    std::vector<dealii::FullMatrix<double>> &matrices) {
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
        }
      }
    }
  }
}

template <int dim, int qdim>
void Transport<dim, qdim>::integrate_boundary_term(
    const std::vector<Ordinate> &ordinates_in_sweep,
    const dealii::FEFaceValues<dim> &fe_face_values,
    std::vector<dealii::FullMatrix<double>> &matrices,
    dealii::BlockVector<double> &src_cell) {
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
      } else {  // inflow
        for (int i = 0; i < fe_face_values.dofs_per_cell; ++i) {
          for (int j = 0; j < fe_face_values.dofs_per_cell; ++j) {
            src_cell.block(n)(i) += -ord_dot_normal
                                    * 1.0 //local_vector_copy(j)
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
    dealii::BlockVector<double> &src_cell) {
  const std::vector<double> &JxW = fe_face_values.get_JxW_values();
  const std::vector<dealii::Tensor<1, dim> > &normals =
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
      } else {  // inflow
        for (int i = 0; i < fe_face_values.dofs_per_cell; ++i) {
          for (int k = 0; k < fe_face_values_neighbor.dofs_per_cell; ++k) {
            std::cout << dst_neighbor.block(n)(k) << std::endl;
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