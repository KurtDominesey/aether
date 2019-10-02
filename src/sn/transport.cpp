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
  }
  octant_directions.resize(std::pow(dim, 2));
  renumberings.resize(
      octant_directions.size(),
      std::vector<dealii::types::global_dof_index>(dof_handler.n_dofs()));
  ordinates_in_octant.resize(octant_directions.size());
  if (dim == 1) {
    octant_directions[0] = dealii::Point<dim>(+1);
    octant_directions[1] = dealii::Point<dim>(-1);
    dealii::DoFRenumbering::compute_downstream(renumberings[0], renumberings[1],
                                               dof_handler,
                                               octant_directions[0], false);
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
    dealii::DoFRenumbering::compute_downstream(renumberings[0], renumberings[2],
                                               dof_handler,
                                               octant_directions[0], false);
    dealii::DoFRenumbering::compute_downstream(renumberings[1], renumberings[3],
                                               dof_handler,
                                               octant_directions[1], false);
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
    dealii::DoFRenumbering::compute_downstream(renumberings[0], renumberings[6],
                                               dof_handler,
                                               octant_directions[0], false);
    dealii::DoFRenumbering::compute_downstream(renumberings[1], renumberings[7],
                                               dof_handler,
                                               octant_directions[1], false);
    dealii::DoFRenumbering::compute_downstream(renumberings[2], renumberings[4],
                                                dof_handler,
                                               octant_directions[2], false);
    dealii::DoFRenumbering::compute_downstream(renumberings[3], renumberings[5],
                                               dof_handler,
                                               octant_directions[3], false);
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
  dof_handler.renumber_dofs(renumberings[0]);
  // setup info_box
  int num_points = dof_handler.get_fe().degree + 1;
  info_box.initialize_gauss_quadrature(num_points, num_points, num_points);
  info_box.initialize_update_flags();
  dealii::UpdateFlags update_flags = dealii::update_quadrature_points |
                                     dealii::update_values |
                                     dealii::update_gradients;
  info_box.add_update_flags(update_flags);
  info_box.initialize(dof_handler.get_fe(), dealii::MappingQ1<dim>());
}

template <int dim, int qdim>
void Transport<dim, qdim>::vmult(dealii::BlockVector<double> &dst,
                                 const dealii::BlockVector<double> &src) {
  dealii::MeshWorker::LoopControl loop_control;
  loop_control.cells_first = false;  // loop over faces first
  int num_octants = octant_directions.size();
  for (int oct = 0; oct < num_octants; ++oct) {
    // dof_handler should first be ordered according to renumberings[0]
    DoFInfo dof_info(dof_handler);
    std::vector<int> &ords = ordinates_in_octant[oct];
    dealii::MeshWorker::loop<dim, dim, DoFInfo,
                             dealii::MeshWorker::IntegrationInfoBox<dim> >(
      dof_handler.begin_active(),
      dof_handler.end(), 
      dof_info, 
      info_box,
      std::bind(&Transport<dim>::integrate_cell_term, this, ords,
                std::placeholders::_1, 
                std::placeholders::_2),
      std::bind(&Transport<dim>::integrate_boundary_term, this, ords, 
                std::placeholders::_1, 
                std::placeholders::_2),
      std::bind(&Transport<dim>::integrate_face_term, this, ords, 
                std::placeholders::_1, 
                std::placeholders::_2, 
                std::placeholders::_3, 
                std::placeholders::_4),
      sweeper,
      loop_control
    );
    int oct_next = (oct + 1 == num_octants) ? 0 : oct + 1;
    dof_handler.renumber_dofs(renumberings[oct_next]);
  }
}

template <int dim, int qdim>
void Transport<dim, qdim>::integrate_cell_term(
    std::vector<int> &octant_to_global, DoFInfo &dinfo, CellInfo &info) {
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
  }
}

template <int dim, int qdim>
void Transport<dim, qdim>::integrate_boundary_term(
    std::vector<int> &octant_to_global, DoFInfo &dinfo, CellInfo &info) {
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
                               * local_vector_copy(j)
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
    std::vector<int> &octant_to_global, DoFInfo &dinfo1, DoFInfo &dinfo2,
    CellInfo &info1, CellInfo &info2) {
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
    dealii::FullMatrix<double> &u1_v1_matrix = dinfo1.matrix(0, false).matrix;
    dealii::FullMatrix<double> &u2_v1_matrix = dinfo1.matrix(0, true).matrix;
    dealii::FullMatrix<double> &u1_v2_matrix = dinfo2.matrix(0, true).matrix;
    dealii::FullMatrix<double> &u2_v2_matrix = dinfo2.matrix(0, false).matrix;
    for (int q = 0; q < fe_face_values.n_quadrature_points; ++q) {
      double ord_dot_normal = ordinates[octant_to_global[n]] * normals[q];
      if (ord_dot_normal > 0) {  // outflow
        for (int i = 0; i < dofs_per_cell; ++i) {
          for (int j = 0; j < dofs_per_cell; ++j) {
            u1_v1_matrix(i, j) += ord_dot_normal
                                  * fe_face_values.shape_value(j, q)
                                  * fe_face_values.shape_value(i, q)
                                  * JxW[q];
          }
        }
        for (int k = 0; k < dofs_per_cell_neighbor; ++k) {
          for (int j = 0; j < dofs_per_cell; ++j) {
            u1_v2_matrix(k, j) += -ord_dot_normal
                                  * fe_face_values.shape_value(j, q)
                                  * fe_face_values_neighbor.shape_value(k, q)
                                  * JxW[q];
          }
        }
      } else {  // inflow
        for (int i = 0; i < dofs_per_cell; ++i) {
          for (int l = 0; l < dofs_per_cell_neighbor; ++l) {
            u2_v1_matrix(i, l) += ord_dot_normal
                                  * fe_face_values_neighbor.shape_value(l, q)
                                  * fe_face_values.shape_value(i, q)
                                  * JxW[q];
          }
        }
        for (int k = 0; k < dofs_per_cell_neighbor; ++k) {
          for (int l = 0; l < dofs_per_cell_neighbor; ++l) {
            u2_v2_matrix(k, l) += -ord_dot_normal
                                  * fe_face_values_neighbor.shape_value(l, q)
                                  * fe_face_values_neighbor.shape_value(k, q)
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