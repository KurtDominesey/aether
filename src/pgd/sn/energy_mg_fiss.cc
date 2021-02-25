#include <deal.II/lac/petsc_full_matrix.h>

#include "pgd/sn/energy_mg_fiss.h"

namespace aether::pgd::sn {

EnergyMgFiss::EnergyMgFiss(const Mgxs &mgxs)
    : EnergyMgFull(mgxs, zero_sources) {}

double EnergyMgFiss::step_eigenvalue(InnerProducts &coefficients) {
  std::vector<std::vector<InnerProducts>> coefficients_matrix =
      {{coefficients}};
  return update(coefficients_matrix);
}

double EnergyMgFiss::update(
    std::vector<std::vector<InnerProducts>> &coefficients, const double tol) {
  AssertDimension(coefficients.size(), modes.size());
  const int num_groups = modes[0].size();
  const int size = modes.size() * num_groups;
  // set matrices
  // dealii::FullMatrix<double> slowing(modes.size() * num_groups);
  // dealii::FullMatrix<double> fission(modes.size() * num_groups);
  dealii::PETScWrappers::FullMatrix slowing(size, size);
  dealii::PETScWrappers::FullMatrix fission(size, size);
  for (int m_row = 0; m_row < modes.size(); ++m_row) {
    int mm_row = m_row * num_groups;
    for (int m_col = 0; m_col < modes.size(); ++m_col) {
      int mm_col = m_col * num_groups;
      for (int g = 0; g < num_groups; ++g) {
        slowing.add(mm_row+g, mm_col+g, coefficients[m_row][m_col].streaming);
        for (int j = 0; j < mgxs.total[g].size(); ++j) {
          slowing.add(mm_row+g, mm_col+g,
              coefficients[m_row][m_col].collision[j] * mgxs.total[g][j]);
          for (int gp = 0; gp < num_groups; ++gp) {
            for (int ell = 0; ell < 1; ++ell) {
              slowing.add(mm_row+g, mm_col+gp, 
                          coefficients[m_row][m_col].scattering[j][ell] 
                          * mgxs.scatter[g][gp][j]);
            }
            fission.add(mm_row+g, mm_col+gp,
                        mgxs.chi[g][j] 
                        * coefficients[m_row][m_col].fission[j] 
                        * mgxs.nu_fission[gp][j]);
          }
        }
      }
    }
  }
  slowing.compress(dealii::VectorOperation::add);
  fission.compress(dealii::VectorOperation::add);
  // set solution vector
  // dealii::Vector<double> solution(modes.size() * num_groups);
  std::vector<dealii::PETScWrappers::MPI::Vector> eigenvectors;
  eigenvectors.emplace_back(MPI_COMM_WORLD, size, size);
  eigenvectors[0].compress(dealii::VectorOperation::add);
  for (int m = 0; m < modes.size(); ++m) {
    int mm = m * num_groups;
    for (int g = 0; g < num_groups; ++g) {
      eigenvectors[0][mm+g] += modes[m][g];
    }
  }
  eigenvectors[0].compress(dealii::VectorOperation::add);
  dealii::PETScWrappers::MPI::Vector ax(eigenvectors[0]);
  dealii::PETScWrappers::MPI::Vector bx(eigenvectors[0]);
  fission.vmult(ax, eigenvectors[0]);
  slowing.vmult(bx, eigenvectors[0]);
  double rayleigh = (eigenvectors[0] * ax) / (eigenvectors[0] * bx);
  std::cout << "rayleigh-e: " << rayleigh << "\n";
  // rayleigh = 0;  // !!
  // eigenvectors[0].compress(dealii::VectorOperation::unknown);
  // make matrices sparse
  // dealii::SparsityPattern pattern_slowing;
  // dealii::SparsityPattern pattern_fission;
  // pattern_slowing.copy_from(slowing);
  // pattern_fission.copy_from(fission);
  // dealii::SparseMatrix<double> slowing_sp(pattern_slowing);
  // dealii::SparseMatrix<double> fission_sp(pattern_fission);
  // slowing_sp.copy_from(slowing);
  // fission_sp.copy_from(fission);
  dealii::SolverControl control(10, std::clamp(tol, 1e-5, 1e-3));
  // dealii::SLEPcWrappers::SolverGeneralizedDavidson eigensolver(control);
  dealii::SLEPcWrappers::SolverJacobiDavidson eigensolver(control);
  eigensolver.set_target_eigenvalue(rayleigh/*+1e-2*/);
  eigensolver.set_which_eigenpairs(EPS_LARGEST_MAGNITUDE);
  if (eigenvectors[0].l2_norm() > 0)
    eigensolver.set_initial_space(eigenvectors);
  // set preconditioner
  bool use_pc = false;
  if (use_pc) {
    // dealii::PETScWrappers::FullMatrix shifted(size, size);
    // shifted.add(1, fission);
    // shifted.add(-rayleigh, slowing);
    dealii::PETScWrappers::SparseMatrix shifted(size, size, size);
    dealii::FullMatrix<double> shifted_full(size);
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        shifted.set(i, j, fission.el(i, j) - rayleigh * slowing.el(i, j));
        shifted_full(i, j) = fission.el(i, j) - rayleigh * slowing.el(i, j);
      }
    }
    shifted.compress(dealii::VectorOperation::insert);
    std::cout << "shifted it\n";
    dealii::PETScWrappers::PreconditionSOR preconditioner(shifted);
    // try dealii operators
    dealii::SparsityPattern pattern;
    pattern.copy_from(shifted_full);
    dealii::SparseMatrix<double> shifted_sparse(pattern);
    shifted_sparse.copy_from(shifted_full);
    using BlockSOR = dealii::PreconditionBlockSOR<dealii::SparseMatrix<double>>;
    BlockSOR shifted_gs;
    shifted_gs.initialize(shifted_sparse, BlockSOR::AdditionalData(num_groups));
    aether::PETScWrappers::MatrixFreeWrapper<BlockSOR> shifted_gs_mf(
        MPI_COMM_WORLD, size, size, size, size, shifted_gs);
    aether::PETScWrappers::PreconditionerShell shifted_gs_pc(shifted_gs_mf);
    // dealii::SolverControl control_dummy(1, 0);
    // dealii::PETScWrappers::SolverPreOnly solver_pc(control_dummy);
    dealii::ReductionControl control_pc(3, 1e-6, 1e-3);
    dealii::PETScWrappers::SolverGMRES solver_pc(control_pc);
    solver_pc.initialize(shifted_gs_pc);
    // solver_pc.initialize(preconditioner);
    aether::SLEPcWrappers::TransformationPreconditioner stprecond(
        MPI_COMM_WORLD, shifted_gs_mf);
    // aether::SLEPcWrappers::TransformationPreconditioner stprecond(
    //     MPI_COMM_WORLD, shifted);
    stprecond.set_matrix_mode(ST_MATMODE_SHELL);
    stprecond.set_solver(solver_pc);
    eigensolver.set_transformation(stprecond);
  }
  // solve eigenproblem
  std::vector<double> eigenvalues;
  try {
    eigensolver.solve(fission, slowing, eigenvalues, eigenvectors);
    if (eigenvalues[0] < 0) {
      eigenvalues[0] *= -1;
      eigenvectors[0] *= -1;
    }
    for (int m = 0, i = 0; m < this->modes.size(); ++m) {
      for (int g = 0; g < num_groups; ++g, ++i) {
        modes[m][g] = eigenvectors[0][i];
      }
    }
    std::cout << "k-updatestep: " << eigenvalues[0] << "\n";
    this->eigenvalue = eigenvalues[0];
    return eigenvalues[0];
  } catch (...) {
    std::cout << "failed\n";
    for (int m = 0, i = 0; m < this->modes.size(); ++m) {
      for (int g = 0; g < num_groups; ++g, ++i) {
        modes[m][g] = eigenvectors[0][i];
      }
    }
    return 0;
  }
}

void EnergyMgFiss::update(
    std::vector<std::vector<InnerProducts>> coefficients_x,
    std::vector<std::vector<double>> coefficients_b) {
  update(coefficients_x);
}

void EnergyMgFiss::set_matrix(InnerProducts coefficients) {
  EnergyMgFull::set_matrix(coefficients);
  const int num_groups = this->mgxs.total.size();
  for (int g = 0; g < num_groups; ++g) {
    for (int gp = 0; gp < num_groups; ++gp) {
      for (int j = 0; j < this->mgxs.total[g].size(); ++j) {
        this->matrix[g][gp] += this->mgxs.chi[g][j]
                               * this->mgxs.nu_fission[gp][j]
                               * coefficients.fission[j]
                               / this->eigenvalue;
      }
    }
  }
}

void EnergyMgFiss::set_source(std::vector<double> coefficients_b, 
                              std::vector<InnerProducts> coefficients_x) {
  EnergyMgFull::set_source(coefficients_b, coefficients_x);
  AssertDimension(coefficients_x.size(), modes.size() - 1);
  for (int m = 0; m < coefficients_x.size(); ++m) {
    for (int g = 0; g < this->mgxs.total.size(); ++g) {
      for (int gp = 0; gp < this->mgxs.total.size(); ++gp) {
        for (int j = 0; j < this->mgxs.total[g].size(); ++j) {
          source[g] -= this->mgxs.chi[g][j]
                        * this->mgxs.nu_fission[gp][j]
                        * coefficients_x[m].fission[j]
                        * modes[m][gp]
                        / this->eigenvalue;
        }
      }
    }
  }
}

}