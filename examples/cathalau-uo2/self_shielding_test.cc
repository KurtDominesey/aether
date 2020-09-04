#ifndef AETHER_EXAMPLES_CATHALAU_SELF_SHIELDING_TEST_H_
#define AETHER_EXAMPLES_CATHALAU_SELF_SHIELDING_TEST_H_

#include "cathalau_test.cc"

#include "sn/fixed_source_problem.h"

namespace cathalau {

using namespace aether;
using namespace aether::sn;

class SelfShieldingTest : public CathalauTest {
 protected:
  using CathalauTest::mesh;
  using CathalauTest::quadrature;
  using CathalauTest::dof_handler;
  using CathalauTest::mgxs;
  const int num_polar = 4;
  const int num_azim = 9;  // must be odd

  void SetUp() override {
    CathalauTest::group_structure = "SHEM-361";
    CathalauTest::SetUp();
    refine_azimuthal(mesh, 2);
    refine_radial(mesh, 2, this->max_levels);
    this->PrintMesh();
    dealii::FE_DGQ<dim_> fe(1);
    dof_handler.initialize(mesh, fe);
    Assert(n_azim % 2, dealii::ExcMessage("Must be odd"));
    quadrature = QPglc<dim_, qdim_>(num_polar, num_azim);
  }

  void Run(const int max_iters, const double tol, 
           const bool precomputed=false) {
    const int j_fuel = 2;  // see CathalauTest::materials
    const int num_groups = mgxs->total.size();
    // set up problem
    std::vector<std::vector<dealii::BlockVector<double>>> 
        boundary_conditions(num_groups);
    FixedSourceProblem<dim_, qdim_> problem(
        dof_handler, quadrature, *mgxs, boundary_conditions);
    dealii::BlockVector<double> flux(
        num_groups, quadrature.size()*dof_handler.n_dofs());
    dealii::BlockVector<double> source(flux);
    // set fission source and find diagonal position
    int i_center = -1;  // dof index at center of fuel
    int i_diag = -1;  // dof index at diagonal position in fuel
    std::vector<dealii::types::global_dof_index> i_diags;
    dealii::FullMatrix<double> mass;
    const int dofs_per_cell = dof_handler.get_fe().dofs_per_cell;
    std::vector<dealii::types::global_dof_index> dof_indices(dofs_per_cell);
    int c = 0;
    for (auto cell = dof_handler.begin_active(); cell != dof_handler.end();
         ++cell, ++c) {
      const int j = cell->material_id();
      cell->get_dof_indices(dof_indices);
      // set fission source
      for (int g = 0; g < num_groups; ++g)
        for (int i = 0; i < dofs_per_cell; ++i)
          for (int n = 0; n < quadrature.size(); ++n)
            source.block(g)[n*dof_handler.n_dofs()+dof_indices[i]] =
                mgxs->chi[g][j];
      // find diagonal position
      if (i_diag != -1)
        continue;  // already found
      for (int v = 0; v < dealii::GeometryInfo<dim_>::vertices_per_cell; ++v) {
        i_diags = dof_indices;
        mass = problem.transport.cell_matrices[c].mass;
        const dealii::Point<dim_> &vertex = cell->vertex(v);
        double tol_geom = 1e-6;
        bool at_diag = 
            std::abs(vertex(0)-radii.front()/std::sqrt(2)) < tol_geom && 
            std::abs(vertex(1)-radii.front()/std::sqrt(2)) < tol_geom;
        bool at_center = vertex(0) == 0 && vertex(1) == 0; 
        if (at_diag && j == j_fuel && i_diag == -1) {
          i_diag = dof_indices[v];
          std::cout << vertex(0) << " " << vertex(1) << " " << i_diag << std::endl;
        }
        if (at_center && i_center == -1) {
          i_center = dof_indices[v];
          std::cout << vertex(0) << " " << vertex(1) << " " << i_center << std::endl;
        }
      }
    }
    // run full order
    const std::string filename = this->GetTestName() + ".h5";
    if (precomputed) {
      dealii::HDF5::File file(
          filename, dealii::HDF5::File::FileAccessMode::open);
      flux = file.open_dataset("flux_full").read<dealii::Vector<double>>();
    } else {
      dealii::BlockVector<double> uncollided(flux);
      problem.sweep_source(uncollided, source);
      dealii::ReductionControl control_wg(50, tol, 1e-2);
      dealii::SolverGMRES<dealii::Vector<double>> solver_wg(control_wg);
      FixedSourceGS<dealii::SolverGMRES<dealii::Vector<double>>, dim_, qdim_>
          preconditioner(problem.fixed_source, solver_wg);
      dealii::SolverControl control(max_iters, tol);
      control.enable_history_data();
      dealii::SolverRichardson<dealii::BlockVector<double>> solver(control);
      solver.connect([](const unsigned int iteration,
                        const double check_value,
                        const dealii::BlockVector<double>&) {
        std::cout << iteration << ": " << check_value << std::endl;
        return dealii::SolverControl::success;
      });
      try {
        solver.solve(problem.fixed_source, flux, uncollided, preconditioner);
      } catch(dealii::SolverControl::NoConvergence &failure) {
        failure.print_info(std::cout);
      }
      this->WriteFlux(flux, control.get_history_data(), filename);
    }
    this->PlotFlux(flux, problem.d2m, mgxs->group_structure, "full");
    // find diagonal spectra
    const int n_out = (num_azim - 1) / 2 + 1;
    const int n_in = n_out + num_azim * 2;
    // casmo-sh-70 coarse group 27
    const int g_min = 224;
    const int g_max = 276;
    const int num_subgroups = g_max - g_min + 1;
    dealii::Vector<double> spectra_out(num_subgroups);
    dealii::Vector<double> spectra_in(num_subgroups);
    dealii::Vector<double> spectra_iso(num_subgroups);
    dealii::Vector<double> spectra_center(num_subgroups);
    dealii::Vector<double> cross_section(num_subgroups);
    Assert(quadrature.is_tensor_product(), dealii::ExcInvalidState());
    const dealii::Quadrature<1> &q_polar = quadrature.get_tensor_basis()[0];
    for (int g = g_min, gg = 0; g <= g_max; ++g, ++gg) {
      cross_section[gg] = mgxs->total[g][j_fuel];
      for (int i = 0; i < dofs_per_cell; ++i) {
        for (int j = 0; j < dofs_per_cell; ++j) {
          i_diag = i_diags[j];
          for (int n_polar = 0; n_polar < num_polar; ++n_polar) {
            int nn_out = n_out * num_polar + n_polar;
            int nn_in = n_in * num_polar + n_polar;
            spectra_out[gg] += flux.block(g)[nn_out*dof_handler.n_dofs()+i_diag]
                              * mass[i][j] * q_polar.weight(n_polar);
            spectra_in[gg] += flux.block(g)[nn_in*dof_handler.n_dofs()+i_diag]
                              * mass[i][j] * q_polar.weight(n_polar);
          }
          for (int n = 0; n < quadrature.size(); ++n){
            spectra_iso[gg] += flux.block(g)[n*dof_handler.n_dofs()+i_diag] 
                               * mass[i][j] * quadrature.weight(n);
          }
        }
      }
      for (int n = 0; n < quadrature.size(); ++n) {
        spectra_center[gg] += flux.block(g)[n*dof_handler.n_dofs()+i_center] 
                              * quadrature.weight(n);
      }
    }
    double vol = 0;
    for (int i = 0; i < dofs_per_cell; ++i) {
      for (int j = 0; j < dofs_per_cell; ++j) {
        vol += mass[i][j];
      }
    }
    for (int g = 0; g < num_subgroups; ++g) {
      spectra_iso[g] /= vol;
      spectra_in[g] /= vol;
      spectra_out[g] /= vol;
    }
    // print out
    std::vector<dealii::Vector<double>*> outputs = 
        {&cross_section, &spectra_iso, &spectra_in, &spectra_out, &spectra_center};
    std::ofstream csv(this->GetTestName()+".csv", std::ofstream::trunc);
    csv << "cross_section, spectra_iso, spectra_in, spectra_out, spectra_center\n";
    csv.precision(16);
    csv << std::scientific;
    for (int g = 0; g < num_subgroups; ++g) {
      for (int c = 0; c < outputs.size(); ++c) {
        if (c > 0)
          csv << ", ";
        csv << (*outputs[c])[g];
      }
      csv << "\n";
    }
  }
};

TEST_F(SelfShieldingTest, Diagonal) {
  this->Run(20, 1e-6, true);
}

}  // namespace cathalau

#endif  // AETHER_EXAMPLES_CATHALAU_SELF_SHIELDING_TEST_H_