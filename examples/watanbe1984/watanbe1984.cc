#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_richardson.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/data_out.h>

#include "types/types.h"
#include "sn/quadrature.cc"
#include "sn/discrete_to_moment.cc"
#include "sn/moment_to_discrete.cc"
#include "sn/transport.cc"
#include "sn/transport_block.cc"
#include "sn/scattering.cc"
#include "sn/scattering_block.cc"
#include "sn/within_group.cc"

using namespace aether;

int main() {
  // Create mesh
  static const int dim = 2;
  dealii::Triangulation<dim> mesh;
  double left = 0;
  double right = 10;
  dealii::GridGenerator::subdivided_hyper_cube(mesh, right/1.25, left, right);
  // Assign materials and boundary conditions
  using Cell = typename dealii::Triangulation<dim>::cell_iterator;
  using Face = typename dealii::Triangulation<dim>::face_iterator;
  for (Cell cell = mesh.begin(); cell != mesh.end(); ++cell) {
    const dealii::Point<dim> &center = cell->center();
    bool in_corner = center[0] < 1.25 && center[1] < 1.25;
    bool in_periphery = center[0] > 0.5 || center[1] > 0.5;
    if (in_corner || in_periphery)
      cell->set_material_id(1);
    else
      cell->set_material_id(0);
    for (int f = 0; f < dealii::GeometryInfo<dim>::faces_per_cell; ++f) {
      if (!cell->at_boundary(f))
        continue;
      Face face = cell->face(f);
      bool at_left_x = face->vertex(0)[0] == left && face->vertex(1)[0] == left;
      bool at_left_y = face->vertex(0)[1] == left && face->vertex(1)[1] == left;
      if (at_left_x || at_left_y)
        face->set_boundary_id(types::reflecting_boundary_id);
    }
  }
  mesh.refine_global(4);
  // Create quadrature
  int order_polar = 8;
  int order_azim = 16;
  dealii::Quadrature<1> q_polar = dealii::QGauss<1>(2 * order_polar);
  if (dim == 2)
    q_polar = sn::impose_polar_symmetry(q_polar);
  dealii::QIterated<1> q_azim(dealii::QMidpoint<1>(), 4 * order_azim);
  dealii::Quadrature<2> quadrature(q_polar, q_azim);
  // Create cross sections
  std::vector<double> xs_total = {0.0, 0.2};
  std::vector<double> xs_scatter = {0.0, 0.19};
  // Create finite elements
  dealii::FE_DGQ<dim> fe(1);
  dealii::DoFHandler dof_handler(mesh);
  dof_handler.distribute_dofs(fe);
  // Create operators
  sn::DiscreteToMoment d2m(quadrature);
  sn::MomentToDiscrete m2d(quadrature);
  sn::Transport transport(dof_handler, quadrature);
  sn::Scattering scattering(dof_handler);
  // Create boundary conditions
  std::vector<dealii::BlockVector<double>> boundary_conditions(
      1, dealii::BlockVector<double>(quadrature.size(), fe.dofs_per_cell));
  // Create group operators
  sn::TransportBlock transport_g(transport, xs_total, boundary_conditions);
  sn::ScatteringBlock scattering_gg(scattering, xs_scatter);
  sn::WithinGroup within_g(transport_g, m2d, scattering_gg, d2m);
  // Initialize solution and source vectors
  dealii::BlockVector<double> flux(quadrature.size(), dof_handler.n_dofs());
  dealii::BlockVector<double> source(flux.get_block_indices());
  dealii::BlockVector<double> uncollided(flux.get_block_indices());
  // Set source vector
  std::vector<dealii::types::global_dof_index> dof_indices(fe.dofs_per_cell);
  using ActiveCell = typename dealii::DoFHandler<dim>::active_cell_iterator;
  for (ActiveCell cell = dof_handler.begin_active(); cell != dof_handler.end();
       ++cell) {
    const dealii::Point<dim> &center = cell->center();
    bool in_corner = center[0] < 1.25 && center[1] < 1.25;
    if (!in_corner)
      continue;
    cell->get_dof_indices(dof_indices);
    for (int n = 0; n < quadrature.size(); ++n)
      for (int i = 0; i < dof_indices.size(); ++i)
        source.block(n)[dof_indices[i]] = 1.0;
  }
  // Solve problem
  dealii::SolverControl solver_control(5000, 1e-8);
  dealii::SolverGMRES<dealii::BlockVector<double>> solver(solver_control);
  within_g.transport.vmult(uncollided, source, false);
  solver.solve(within_g, flux, uncollided, dealii::PreconditionIdentity());
  // Post-process
  std::cout << "iterations: " << solver_control.last_step() << std::endl;
  bool positive = true;
  for (int n = 0; n < quadrature.size(); ++n) {
    for (int i = 0; i < dof_handler.n_dofs(); ++i) {
      double value = flux.block(n)[i];
      if (value < 0) {
        std::cout << n << std::endl;
        positive = false;
        break;
      }
    }
  }
  std::cout << "all positive? "  << (positive ? "true" : "false") << std::endl;
  dealii::BlockVector<double> flux_m(1, dof_handler.n_dofs());
  d2m.vmult(flux_m, flux);
  // Output results
  dealii::DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(flux_m, "flux");
  // for (int n = 0; n < quadrature.size(); ++n) {
  //   data_out.add_data_vector(flux.block(n), "n" + std::to_string(n));
  // }
  data_out.build_patches();
  std::ofstream output("flux.vtu");
  data_out.write_vtu(output);
}