#include <deal.II/grid/tria.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/hdf5.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_relaxation.h>
#include <deal.II/lac/solver_richardson.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_dof_data.h>

#include "mesh/mesh.hpp"
#include "types/types.hpp"
#include "sn/quadrature.cpp"
#include "sn/discrete_to_moment.cpp"
#include "sn/moment_to_discrete.cpp"
#include "sn/transport.cpp"
#include "sn/transport_block.cpp"
#include "sn/scattering.cpp"
#include "sn/scattering_block.cpp"
#include "sn/within_group.cpp"
#include "sn/fixed_source.cpp"
#include "sn/fixed_source_gs.cpp"

using namespace aether;

int main() {
  // Create mesh
  std::cout << "create mesh\n";
  static const int dim = 2;
  dealii::Triangulation<dim> mesh;
  double pitch = 0.63;  // cm
  double radius = 0.54;  // cm
  mesh_quarter_pincell(mesh, {radius}, pitch, {1, 0});
  mesh.refine_global(2);
  // Create finite elements
  std::cout << "create finite elements\n";
  dealii::FE_DGQ<dim> fe(1);
  dealii::DoFHandler dof_handler(mesh);
  dof_handler.distribute_dofs(fe);
  // Create quadrature
  std::cout << "create quadrature\n";
  int num_ords_per_oct = 8;
  dealii::Quadrature<1> q_polar = dealii::QGauss<1>(num_ords_per_oct*2);
  std::vector<dealii::Point<1>> points(num_ords_per_oct);
  std::vector<double> weights(num_ords_per_oct);
  for (int n = 0; n < num_ords_per_oct; ++n) {
    points[n](0) = (n + 0.5) / num_ords_per_oct;
    weights[n] = 1.0 / num_ords_per_oct;
  }
  dealii::Quadrature<1> q_azimuthal_base(points, weights);
  dealii::QIterated<1> q_azimuthal(q_azimuthal_base, 4);
  double sum_weights = 0;
  for (int n = 0; n < q_azimuthal.size(); ++n)
    sum_weights += q_azimuthal.weight(n);
  AssertThrow(std::abs(1 - sum_weights) < 1e-13,
      dealii::ExcMessage("Azimuthal weights sum to " +
                         std::to_string(sum_weights) + " not one"));
  q_polar = sn::impose_polar_symmetry(q_polar);
  dealii::QAnisotropic<2> quadrature(q_polar, q_azimuthal);
  std::ofstream quadrature_csv("quadrature.csv");
  for (int n = 0; n < quadrature.size(); ++n)
    quadrature_csv << quadrature.point(n)[0] << ","
                   << quadrature.point(n)[1] << ","
                   << quadrature.weight(n) << "\n";
  quadrature_csv.close();
  // Read cross-section files
  std::cout << "read cross-section files\n";
  namespace hdf5 = dealii::HDF5;
  hdf5::File mgxs("c5g7.h5", hdf5::File::FileAccessMode::open);
  std::string temperature = "294K";
  std::vector<std::string> materials = {"water", "uo2"};
  int num_groups = mgxs.get_attribute<int>("energy_groups");
  std::vector<std::vector<double>> xs_total_pivot(
      materials.size(), std::vector<double>(num_groups));
  std::vector<std::vector<std::vector<double>>> xs_scatter_pivot(
      materials.size(), std::vector<std::vector<double>>(
                            num_groups, std::vector<double>(num_groups)));
  std::vector<std::vector<double>> chi = xs_total_pivot;
  for (int i = 0; i < materials.size(); ++i) {
    hdf5::Group material = mgxs.open_group(materials[i]);
    hdf5::Group library = material.open_group(temperature);
    xs_total_pivot[i] =
        library.open_dataset("total").read<std::vector<double>>();
    if (material.get_attribute<bool>("fissionable"))
      chi[i] = library.open_dataset("chi").read<std::vector<double>>();
    hdf5::Group scatter_data = library.open_group("scatter_data");
    std::vector<int> g_min = 
        scatter_data.open_dataset("g_min").read<std::vector<int>>();
    std::vector<int> g_max = 
        scatter_data.open_dataset("g_max").read<std::vector<int>>();
    std::vector<double> scatter_matrix =
        scatter_data.open_dataset("scatter_matrix").read<std::vector<double>>();
    int gg = 0;
    for (int ga = 0; ga < num_groups; ++ga)
      for (int gb = g_min[ga]; gb <= g_max[ga]; ++gb, ++gg)
        xs_scatter_pivot[i][ga][gb-1] = scatter_matrix[gg];
    AssertDimension(gg, scatter_matrix.size());
  }
  // Pivot cross-section data
  std::cout << "pivot cross-section data\n";
  std::vector<std::vector<double>> xs_total(
      num_groups, std::vector<double>(materials.size()));
  std::vector<std::vector<std::vector<double>>> xs_scatter(
      num_groups, xs_total);
  for (int i = 0; i < materials.size(); ++i) {
    for (int g = 0; g < num_groups; ++g) {
      xs_total[g][i] = xs_total_pivot[i][g];
      // xs_total[g][i] = 1;
      // xs_scatter[g][g][i] = 0.9;
      for (int gp = 0; gp < num_groups; ++gp)
        xs_scatter[g][gp][i] = xs_scatter_pivot[i][gp][g];
    }
  }
  // Create boundary conditions (empty, all reflecting)
  std::cout << "create boundary condtions\n";
  // std::vector<dealii::BlockVector<double>> boundary_conditions(
  //     1, dealii::BlockVector<double>(quadrature.size(), fe.dofs_per_cell));
  std::vector<dealii::BlockVector<double>> boundary_conditions;
  // Create operators
  std::cout << "create operators\n";
  sn::DiscreteToMoment d2m(quadrature);
  sn::MomentToDiscrete m2d(quadrature);
  sn::Transport transport(dof_handler, quadrature);
  sn::Scattering scattering(dof_handler);
  // Specialize operators to groups
  std::cout << "specialize operators to groups\n";
  std::vector<sn::WithinGroup<dim>> within_groups;
  std::vector<std::vector<sn::ScatteringBlock<dim>>> downscattering(num_groups);
  std::vector<std::vector<sn::ScatteringBlock<dim>>> upscattering(num_groups); 
  for (int g = 0; g < num_groups; ++g) {
    sn::TransportBlock transport_wg(transport, xs_total[g], 
                                    boundary_conditions);
    sn::ScatteringBlock scattering_wg(scattering, xs_scatter[g][g]);
    within_groups.emplace_back(transport_wg, m2d, scattering_wg, d2m);
    AssertDimension(xs_scatter[g].size(), num_groups);
    for (int gp = g - 1; gp >= 0; --gp)
      downscattering[g].emplace_back(scattering, xs_scatter[g][gp]);
    for (int gp = g + 1; gp < num_groups; ++gp)
      upscattering[g].emplace_back(scattering, xs_scatter[g][gp]);
  }
  for (int gp = 0; gp < num_groups; ++gp) {
    for (int g = 0; g < num_groups; ++g)
      std::cout << xs_scatter[g][gp][1] << " ";
    std::cout << "\n";
  }
  // Create final (fixed source) operators
  std::cout << "create final (fixed source) operators\n";
  sn::FixedSource fixed_source(within_groups, downscattering, upscattering, 
                               m2d, d2m);
  dealii::ReductionControl control_wg(3000, 1e-5, 1e-5);
  dealii::SolverGMRES<dealii::Vector<double>> solver_wg(control_wg);
  sn::FixedSourceGS fixed_source_gs(within_groups, downscattering, upscattering,
                                    m2d, d2m, solver_wg);
  // Initialize storage
  std::cout << "initialize storage\n";
  int size = quadrature.size() * dof_handler.n_dofs();
  dealii::BlockVector<double> flux(num_groups, size);
  dealii::BlockVector<double> source(num_groups, size);
  dealii::BlockVector<double> uncollided(num_groups, size);
  // Set source and boundary ids
  std::cout << "set source and boundary ids\n";
  std::vector<dealii::types::global_dof_index> dof_indices(fe.dofs_per_cell);
  using Cell = typename dealii::DoFHandler<dim>::active_cell_iterator;
  using Face = typename dealii::DoFHandler<dim>::active_face_iterator;
  for (Cell cell = dof_handler.begin_active(); cell != dof_handler.end();
       ++cell) {
    // set source
    cell->get_dof_indices(dof_indices);
    for (int g = 0; g < num_groups; ++g)
      for (int n = 0; n < quadrature.size(); ++n)
        for (int i = 0; i < dof_indices.size(); ++i)
          source.block(g)[dof_indices[i] + n*dof_handler.n_dofs()]
            = chi[cell->material_id()][g];
            // = cell->material_id(); //chi[1][g];
    // set boundary ids
    if (cell->at_boundary()) {
      for (int f = 0; f < dealii::GeometryInfo<dim>::faces_per_cell; ++f) {
        Face face = cell->face(f);
        if (face->at_boundary())
          face->set_boundary_id(types::reflecting_boundary_id);
      }
    }
  }
  // Sweep source
  std::cout << "sweep source\n";
  for (int g = 0; g < num_groups; ++g)
    within_groups[g].transport.vmult(uncollided.block(g), source.block(g), 
                                     false);
  // Solve the fixed source problem
  std::cout << "solve the fixed source problem\n";
  dealii::SolverControl control_mg(5000, 1e-4);
  dealii::SolverRichardson<dealii::BlockVector<double>> solver_mg(control_mg);
  solver_mg.solve(fixed_source, flux, uncollided, fixed_source_gs);
  // try {
  //   solver_mg.solve(fixed_source, flux, uncollided, dealii::PreconditionIdentity());
  // } catch (dealii::SolverControl::NoConvergence &e) {
  //   std::cout << e.what();
  // }
  // Output results
  std::cout << "output results\n";
  dealii::DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  dealii::BlockVector<double> flux_m(num_groups, dof_handler.n_dofs());
  dealii::BlockVector<double> source_m(num_groups, dof_handler.n_dofs());
  for (int g = 0; g < num_groups; ++g) {
    d2m.vmult(flux_m.block(g), flux.block(g));
    data_out.add_data_vector(
        flux_m.block(g), "g" + std::to_string(g),
        dealii::DataOut_DoFData<dealii::DoFHandler<dim>,
                                dim>::DataVectorType::type_dof_data);
    // d2m.vmult(source_m.block(g), source.block(g));
    // data_out.add_data_vector(source_m.block(g), "q"+std::to_string(g));
  }
  data_out.build_patches();
  std::ofstream output("flux.vtu");
  data_out.write_vtu(output);
}