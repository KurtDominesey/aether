#include "problem.h"

Problem::Problem(const int &num_cells, 
                 const int &num_ords, 
                 const double &length) {
  const int num_groups = 3;
  mgxs = std::make_unique<aether::Mgxs>(num_groups, 1, 1);
  mgxs->nu_fission[0][0] = 3.0 * 0.006;
  mgxs->nu_fission[1][0] = 2.5 * 0.06;
  mgxs->nu_fission[2][0] = 2.0 * 0.9;
  mgxs->total[0][0] = 0.240;
  mgxs->total[1][0] = 0.975;
  mgxs->total[2][0] = 3.1;
  mgxs->chi[0][0] = 0.96;
  mgxs->chi[1][0] = 0.04;
  mgxs->chi[2][0] = 0.;
  mgxs->scatter[0][0][0] = 0.024;
  mgxs->scatter[1][0][0] = 0.171;
  mgxs->scatter[2][0][0] = 0.033;
  mgxs->scatter[0][1][0] = 0.;
  mgxs->scatter[1][1][0] = 0.6;
  mgxs->scatter[2][1][0] = 0.275;
  mgxs->scatter[0][2][0] = 0.;
  mgxs->scatter[1][2][0] = 0.;
  mgxs->scatter[2][2][0] = 2.;
  const int dim = 1;
  dealii::FE_DGQ<dim> fe(1);
  dealii::GridGenerator::subdivided_hyper_cube(mesh, num_cells, 0, 2*length);
  dof_handler.initialize(mesh, fe);
  quadrature = std::make_unique<aether::sn::QPglc<dim>>(num_ords);
  // vacuum boundary conditions (for 3 group and 2 boundaries)
  const int num_boundaries = 2;
  boundary_conditions.resize(num_groups, 
      std::vector<dealii::BlockVector<double>>(num_boundaries,
        dealii::BlockVector<double>(quadrature->size(), fe.dofs_per_cell)));
  problem = std::make_unique<aether::sn::FissionProblem<dim>>(
      dof_handler, *quadrature, *mgxs, boundary_conditions);
}

int Problem::run_fixed_source() {
  const int num_groups = this->mgxs->total.size();
  dealii::ReductionControl control_wg(100, 1e-6, 1e-2);
  dealii::SolverGMRES<dealii::Vector<double>> solver_wg(control_wg);
  aether::sn::FixedSourceGS fixed_source_gs(this->problem->fixed_source, 
                                            solver_wg);
  dealii::SolverControl control(100, 1e-5);
  dealii::SolverGMRES<dealii::BlockVector<double>> solver(control);
  const int size = this->dof_handler.n_dofs() * this->quadrature->size();
  dealii::BlockVector<double> flux(num_groups, size);
  dealii::BlockVector<double> source(num_groups, size);
  for (int g = 0; g < num_groups; ++g) {
    dealii::Vector<double> scalar(this->dof_handler.n_dofs());
    Source<1, double> func(1., this->mgxs->scatter[g][0][0], 1.);
    dealii::VectorTools::interpolate(this->dof_handler, func, scalar);
    for (int n = 0; n < this->quadrature->size(); ++n) {
        int nn = n * this->dof_handler.n_dofs();
        for (int i = 0 ; i < this->dof_handler.n_dofs(); ++i) {
          source.block(g)[nn+i] = scalar[i];
      }
    }
  }
  solver.solve(this->problem->fixed_source, flux, source, 
               fixed_source_gs);
  std::cout << "FIXED\n";
  this->print_currents(flux);
  this->plot(flux, "FIXED");
}

int Problem::run_criticality() {
  const int num_groups = this->mgxs->total.size();
  dealii::ReductionControl control_wg(100, 1e-6, 1e-2);
  dealii::SolverGMRES<dealii::Vector<double>> solver_wg(control_wg);
  aether::sn::FixedSourceGS fixed_source_gs(this->problem->fixed_source, 
                                            solver_wg);
  dealii::ReductionControl control_fs(100, 1e-6, 1e-2);
  dealii::SolverGMRES<dealii::BlockVector<double>> solver_fs(control_fs);
  aether::sn::FissionSource fission_source(
      this->problem->fixed_source, this->problem->fission, solver_fs, 
      fixed_source_gs);
  dealii::SolverControl control(100, 1e-5);
  const int size = this->dof_handler.n_dofs() * this->quadrature->size();
  dealii::GrowingVectorMemory<dealii::BlockVector<double>> memory;
  dealii::EigenPower<dealii::BlockVector<double>> eigensolver(control, memory);
  double k = 1.0;  // bad initial guess
  dealii::BlockVector<double> flux(num_groups, size);
  flux = 1;
  flux /= flux.l2_norm();
  eigensolver.solve(k, fission_source, flux);
  std::cout << "EIGEN\n";
  std::cout << k << ", " << flux.l2_norm() << "\n";
  this->print_currents(flux);
  this->plot(flux, "EIGEN");
}

void Problem::plot(const dealii::BlockVector<double> &flux, 
                   const std::string &suffix) const {
  const int num_groups = this->mgxs->total.size();
  dealii::DataOut<this->dim> data_out;
  data_out.attach_dof_handler(this->dof_handler);
  std::vector<dealii::Vector<double>> fluxes(num_groups,
      dealii::Vector<double>(this->dof_handler.n_dofs()));
  // quick and sloppy way to compute scalar fluxes
  for (int g = 0; g < num_groups; ++g) {
    data_out.add_data_vector(fluxes[g], "g"+std::to_string(g));
    int gg = g * this->quadrature->size() * this->dof_handler.n_dofs();
    for (int n = 0; n < this->quadrature->size(); ++n) {
      int nn = n * this->dof_handler.n_dofs();
      for (int i = 0; i < this->dof_handler.n_dofs(); ++i) {
        fluxes[g][i] += this->quadrature->weight(n) * flux[gg+nn+i];
      }
    }
  }
  data_out.build_patches();
  std::ofstream output("urr-3-0-sl_"+suffix+".vtu");
  data_out.write_vtu(output);
}

void Problem::print_currents(const dealii::BlockVector<double> &flux) {
  const int num_groups = this->mgxs->total.size();
  const int num_ords = this->quadrature->size();
  const int n_2 = num_ords / 2;
  for (int g = 0; g < num_groups; ++g) {
    double reflected = 0;
    double transmitted = 0;
    for (int n = 0; n < n_2; ++n) {
      const int nn = this->quadrature->size() - 1 - n;
      AssertThrow(this->quadrature->angle(n)[0] < 0, dealii::ExcInvalidState());
      AssertThrow(this->quadrature->angle(nn)[0] > 0, dealii::ExcInvalidState());
      reflected += this->quadrature->weight(n) * 
                   flux.block(g)[nn*this->dof_handler.n_dofs()];
      transmitted += this->quadrature->weight(nn) *
                     flux.block(g)[(n+1)*this->dof_handler.n_dofs()-1];
    }
    std::cout << g << ": " << reflected << ", " << transmitted << "\n";
  }
}