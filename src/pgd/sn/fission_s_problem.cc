#include "pgd/sn/fission_s_problem.h"

namespace aether::pgd::sn {

template <int dim, int qdim>
FissionSProblem<dim, qdim>::FissionSProblem(
    const dealii::DoFHandler<dim> &dof_handler,
    const aether::sn::QAngle<dim, qdim> &quadrature,
    const Mgxs &mgxs,
    const std::vector<std::vector<dealii::BlockVector<double>>>
      &boundary_conditions,
    const int num_modes)
    : FixedSourceSProblem<dim, qdim>(dof_handler, quadrature, mgxs, boundary_conditions, 
                                     num_modes),
      emission(num_modes),
      production(num_modes),
      fission_s(this->transport, this->m2d, emission, production, this->d2m),
      fission_s_gs(this->transport, this->scattering, this->m2d, this->d2m,
                   this->fixed_source_s.streaming, mgxs, boundary_conditions) {
  const int num_groups = mgxs.total.size();
  const int num_materials = mgxs.total[0].size();
  for (int m = 0; m < num_modes; ++m) {
    for (int mp = 0; mp < num_modes; ++mp) {
      emission[m].emplace_back(dof_handler, this->mgxs_pseudos[m][mp].chi);
      production[m].emplace_back(dof_handler, this->mgxs_pseudos[m][mp].nu_fission);
      for (int g = 0; g < num_groups; ++g) {
        for (int j = 0; j < num_materials; ++j) {
          this->mgxs_pseudos[m][mp].chi[g][j] = this->mgxs.chi[g][j];
        }
      }
    }
  }
}

template <int dim, int qdim>
void FissionSProblem<dim, qdim>::set_cross_sections(
    const std::vector<std::vector<InnerProducts>> &coefficients) {
  FixedSourceSProblem<dim, qdim>::set_cross_sections(coefficients);
  const int num_modes = coefficients.size();
  const int num_groups = this->mgxs.total.size();
  const int num_materials = this->mgxs.total[0].size();
  for (int m = 0; m < num_modes; ++m) {
    for (int mp = 0; mp < num_modes; ++mp) {
      for (int g = 0; g < num_groups; ++g) {
        for (int j = 0; j < num_materials; ++j) {
          this->mgxs_pseudos[m][mp].nu_fission[g][j] = 
              this->mgxs.nu_fission[g][j] * coefficients[m][mp].fission[j];
        }
      }
    }
  }
  fission_s_gs.set_cross_sections(this->mgxs_pseudos);
}

template <int dim, int qdim>
double FissionSProblem<dim, qdim>::step(
    dealii::BlockVector<double> &modes,
    const std::vector<std::vector<InnerProducts>> &coefficients,
    const double shift) {
  set_cross_sections(coefficients);
  // if (shift == -1)
    return step_gd(modes, shift);
  // return step_power_shift(modes, shift);
}

template <int dim, int qdim>
double FissionSProblem<dim, qdim>::step_gd(dealii::BlockVector<double> &modes,
                                           const double shift) {
  // wrap operators for use with petsc
  const int block_size = modes.block(0).size();
  aether::PETScWrappers::BlockWrapper fixed_source_s_petsc(
      modes.n_blocks(), MPI_COMM_WORLD, block_size, block_size, 
      this->fixed_source_s);
  aether::PETScWrappers::BlockWrapper fission_s_petsc(
      modes.n_blocks(), MPI_COMM_WORLD, block_size, block_size, fission_s);
  aether::PETScWrappers::BlockWrapper fission_s_gs_petsc(
      modes.n_blocks(), MPI_COMM_WORLD, block_size, block_size, fission_s_gs);
  // get initial rayleigh quotient
  dealii::BlockVector<double> ax(modes);
  dealii::BlockVector<double> bx(modes);
  this->fission_s.vmult(ax, modes);
  this->fixed_source_s.vmult(bx, modes);
  double rayleigh = (modes * ax) / (modes * bx);
  std::cout << "rayleigh? " << rayleigh << "\n";
  fission_s_gs.set_shift(rayleigh);  // shift preconditioner
  // set up eigensolver
  dealii::SolverControl control(20, shift);
  // dealii::SLEPcWrappers::SolverJacobiDavidson eigensolver(control);
  dealii::SLEPcWrappers::SolverGeneralizedDavidson eigensolver(control);
  eigensolver.set_target_eigenvalue(rayleigh);
  // set intial guess
  std::vector<dealii::PETScWrappers::MPI::Vector> eigenvectors;
  eigenvectors.emplace_back(MPI_COMM_WORLD, modes.size(), modes.size());
  for (int i = 0; i < modes.size(); ++i)
    eigenvectors[0][i] = modes[i];
  eigenvectors[0].compress(dealii::VectorOperation::insert);
  eigensolver.set_initial_space(eigenvectors);
  // set up preconditioner
  dealii::SolverControl control_dummy(1, 0);  // dummy, solver ignores this
  dealii::PETScWrappers::SolverPreOnly solver_pc(control_dummy);
  // dealii::ReductionControl control_pc(5, 1e-8, 1e-2);
  // dealii::PETScWrappers::SolverGMRES solver_pc(control_pc);
  aether::PETScWrappers::PreconditionerShell pc(fission_s_gs_petsc);
  solver_pc.initialize(pc);
  aether::SLEPcWrappers::TransformationPreconditioner stprecond(
      MPI_COMM_WORLD, fission_s_gs_petsc);
  stprecond.set_matrix_mode(ST_MATMODE_SHELL);
  stprecond.set_solver(solver_pc);
  eigensolver.set_transformation(stprecond);
  // solve the eigenproblem
  std::vector<double> eigenvalues;
  eigensolver.solve(fission_s_petsc, fixed_source_s_petsc, eigenvalues, 
                    eigenvectors);
  double norm = modes.l2_norm();
  for (int i = 0; i < modes.size(); ++i)
    modes[i] = eigenvectors[0][i] * norm;
  // compute residual
  this->fixed_source_s.vmult(ax, modes);
  fission_s.vmult(bx, modes);
  ax.add(-eigenvalues[0], bx);
  double residual = ax.l2_norm() / modes.l2_norm();
  std::cout << "spatio-angular k: " << eigenvalues[0] << "\n";
  std::cout << "spatio-angular residual: " << residual << "\n";
  for (int i = 0; i < eigenvalues.size(); ++i) {
    std::cout << eigenvalues[i] << " ";
  }
  std::cout << "\n";
  return residual;
}

template <int dim, int qdim>
double FissionSProblem<dim, qdim>::step_power_shift(
    dealii::BlockVector<double> &modes, const double shift) {
  // get initial rayleigh quotient
  dealii::BlockVector<double> ax(modes);
  dealii::BlockVector<double> tmp(modes);  // temporary vector, bx here
  fission_s.vmult(ax, modes);
  this->fixed_source_s.vmult(tmp, modes);
  double rayleigh = (modes * ax) / (modes * tmp);
  std::cout << "rayleigh? " << rayleigh << "\n";
  // one shifted power iteration
  dealii::IterationNumberControl control(10, 1e-10);
  using AdditionalData = 
      dealii::SolverGMRES<dealii::BlockVector<double>>::AdditionalData;
  dealii::SolverRichardson<dealii::BlockVector<double>> solver(control/*,
      AdditionalData(1.2)*/);
  double shift_or_rayleigh = shift == 0 ? rayleigh : shift;
  if (true) {
    tmp = modes;
    for (int i = 0; i < 1; ++i) {
      if (shift == -1)
        ax = modes;
      solver.solve(this->fixed_source_s, modes, ax, this->fixed_source_s_gs);
      // modes.add(-shift_or_rayleigh, tmp);
      fission_s.vmult(ax, modes);
      // modes /= this->l2_norm(modes);
      // modes /= modes.l2_norm();
      // modes /= shift == 0 ? rayleigh : shift;
      // modes /= rayleigh;
    }
  } else {  // sinvert
    shift_or_rayleigh += 5e-3;
    shift_or_rayleigh = 1 / shift_or_rayleigh;
    ShiftedS<dim, qdim> shifted(fission_s, this->fixed_source_s);
    shifted.shift = shift_or_rayleigh;
    fission_s_gs.set_shift(shift_or_rayleigh);
    solver.solve(shifted, modes, tmp, fission_s_gs);
  }
  // modes /= modes.l2_norm();
  // compute new rayleigh quotient
  fission_s.vmult(ax, modes);
  this->fixed_source_s.vmult(tmp, modes);
  double rayleigh_new = (modes * ax) / (modes * tmp);
  std::cout << "new rayleigh: " << rayleigh_new << "\n";
  // compute residual
  ax.add(-(shift == 0 ? rayleigh : shift), tmp);
  double residual = ax.l2_norm(); // / modes.l2_norm();
  std::cout << "spatio-angular residual: " << residual << "\n";
  return residual;
}

template <int dim, int qdim>
void FissionSProblem<dim, qdim>::residual(
    dealii::Vector<double> &residual,
    const dealii::Vector<double> &modes,
    const double k_eigenvalue,
    const std::vector<std::vector<InnerProducts>> &coefficients) {
  const int num_modes = coefficients.size();
  AssertDimension(num_modes, coefficients[0].size());
  const int size = residual.size() / num_modes;
  AssertDimension(residual.size(), modes.size());
  dealii::BlockVector<double> ax(num_modes, size);
  dealii::BlockVector<double> bx(num_modes, size);
  dealii::BlockVector<double> modes_b(num_modes, size);
  modes_b = modes;
  this->set_cross_sections(coefficients);
  fission_s.vmult(ax, modes_b);
  this->fixed_source_s.vmult(bx, modes_b);
  ax.add(-k_eigenvalue, bx);
  residual = ax;
}

template <int dim, int qdim>
void FissionSProblem<dim, qdim>::get_inner_products(
    const dealii::BlockVector<double> &modes,
    std::vector<std::vector<InnerProducts>> &inner_products) {
  this->fixed_source_s.get_inner_products_lhs(inner_products, modes);
}

template <int dim, int qdim>
void FissionSProblem<dim, qdim>::get_inner_products(
    const dealii::Vector<double> &modes,
    std::vector<std::vector<InnerProducts>> &inner_products) {
  const int num_modes = inner_products.size();
  const int size = modes.size() / num_modes;
  dealii::BlockVector<double> modes_b(num_modes, size);
  modes_b = modes;
  get_inner_products(modes_b, inner_products);
}

template <int dim, int qdim>
void FissionSProblem<dim, qdim>::solve_fixed_k(
    dealii::Vector<double> &dst,
    const dealii::Vector<double> &src,
    const double k_eigenvalue,
    const std::vector<std::vector<InnerProducts>> &coefficients) {
  // make block vectors
  const int num_modes = coefficients.size();
  AssertDimension(num_modes, coefficients[0].size());
  const int size = dst.size() / num_modes;
  AssertDimension(dst.size(), size * num_modes);
  AssertDimension(dst.size(), src.size());
  dealii::BlockVector<double> dst_b(num_modes, size);
  dealii::BlockVector<double> src_b(num_modes, size);
  dst_b = src;
  src_b = src;
  const double shift = k_eigenvalue;
  set_cross_sections(coefficients);
  int system = 0;
  switch (system) {
    case 0: {  // A-kB
      // set up operators
      ShiftedS<dim, qdim> shifted(fission_s, this->fixed_source_s);
      shifted.shift = shift;
      fission_s_gs.set_shift(shift);
      // fission_s_gs.vmult(dst_b, src_b);
      // solve system
      dealii::IterationNumberControl control(5, 0);
      dealii::SolverFGMRES<dealii::BlockVector<double>> solver(control);
      solver.solve(shifted, dst_b, src_b, fission_s_gs);
      break;
    }
    case 1: {  // -kB
    // solve system
    // dealii::IterationNumberControl control(5, 0);
    // dealii::SolverFGMRES<dealii::BlockVector<double>> solver(control);
    // solver.solve(this->fixed_source_s, dst_b, src_b, this->fixed_source_s_gs);
    this->fixed_source_s_gs.vmult(dst_b, src_b);
    dst_b /= -k_eigenvalue;
    }
  }
  dst = dst_b;
}

template class FissionSProblem<1>;
template class FissionSProblem<2>;
template class FissionSProblem<3>;

}  // namespace aether::pgd::sn