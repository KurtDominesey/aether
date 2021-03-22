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
  // dealii::SolverControl control(10, std::clamp(tol, 1e-5, 1e-3));
  dealii::SolverControl control(10, 1e-5);
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

void EnergyMgFiss::residual(
    dealii::Vector<double> &residual,
    const dealii::Vector<double> &modes_,
    const double k_eigenvalue,
    const std::vector<std::vector<InnerProducts>> &coefficients) {
  const int num_modes = coefficients.size();
  AssertDimension(num_modes, coefficients[0].size());
  const int num_groups = residual.size() / num_modes;
  AssertDimension(residual.size(), modes_.size())
  dealii::FullMatrix<double> a_minus_kb(residual.size());
  set_fixed_k_matrix(a_minus_kb, k_eigenvalue, coefficients);
  a_minus_kb.vmult(residual, modes_);
}

void EnergyMgFiss::solve_fixed_k(
    dealii::Vector<double> &dst,
    const dealii::Vector<double> &src,
    const double k_eigenvalue,
    const std::vector<std::vector<InnerProducts>> &coefficients) {
  const int num_modes = coefficients.size();
  AssertDimension(num_modes, coefficients[0].size());
  const int num_groups = dst.size() / num_modes;
  AssertDimension(dst.size(), src.size());
  dealii::FullMatrix<double> a_minus_kb(dst.size());
  set_fixed_k_matrix(a_minus_kb, k_eigenvalue, coefficients);
  const int last = dst.size() - 1;
  a_minus_kb.set(last, last, 1);
  dealii::IterationNumberControl control(50, 1e-6);
  dealii::SolverGMRES<dealii::Vector<double>> solver(control);
  // set up preconditioner
  dealii::SparsityPattern pattern;
  pattern.copy_from(a_minus_kb);
  dealii::SparseMatrix<double> a_minus_kb_sp(pattern);
  a_minus_kb_sp.copy_from(a_minus_kb);
  // dealii::PreconditionBlockSOR<dealii::SparseMatrix<double>> preconditioner;
  dealii::PreconditionSOR<dealii::SparseMatrix<double>> preconditioner;
  preconditioner.initialize(a_minus_kb_sp/*,
      dealii::PreconditionBlock<dealii::SparseMatrix<double>>::AdditionalData(
        num_groups)*/);
  // set up normality condition
  // (after setting up preconditioner, because of the zero diagonal)
  a_minus_kb.set(last, last, 0);
  for (int m = 0; m < num_modes; ++m)
    for (int g = 0; g < num_groups; ++g)
      a_minus_kb.set(last, m*num_groups+g, 1e2*2*modes[m][g]);
  // if (true) {  // the smart way
  // for (int m_row = 0; m_row < num_modes; ++m_row) {
  //   int mm_row = m_row * num_groups;
  //   for (int m_col = 0; m_col < num_modes; ++m_col) {
  //     int mm_col = m_col * num_groups;
  //     for (int g = 0; g < num_groups; ++g) {
  //       a_minus_kb.add(mm_row+g, last,
  //           -1 * coefficients[m_row][m_col].streaming * modes[m_col][g]);
  //       for (int j = 0; j < mgxs.total[g].size(); ++j) {
  //         a_minus_kb.add(mm_row+g, last, 
  //             -1 * coefficients[m_row][m_col].collision[j] 
  //             * mgxs.total[g][j]
  //             * modes[m_col][g]);
  //         for (int gp = 0; gp < num_groups; ++gp) {
  //           for (int ell = 0; ell < 1; ++ell) {
  //             a_minus_kb.add(mm_row+g, last, 
  //                            -1 * coefficients[m_row][m_col].scattering[j][ell] 
  //                            * mgxs.scatter[g][gp][j]
  //                            * modes[m_col][gp]);
  //           }
  //         }
  //       }
  //     }
  //   }
  // }
  // } else {  // the other way
  const int size = num_modes * num_groups;
  dealii::FullMatrix<double> slowing(size, size);
  for (int m_row = 0; m_row < modes.size(); ++m_row) {
    int mm_row = m_row * num_groups;
    for (int m_col = 0; m_col < modes.size(); ++m_col) {
      int mm_col = m_col * num_groups;
      for (int g = 0; g < num_groups; ++g) {
        slowing.set(mm_row+g, mm_col+g, coefficients[m_row][m_col].streaming);
        for (int j = 0; j < mgxs.total[g].size(); ++j) {
          slowing.set(mm_row+g, mm_col+g,
              coefficients[m_row][m_col].collision[j] * mgxs.total[g][j]);
          for (int gp = 0; gp < num_groups; ++gp) {
            for (int ell = 0; ell < 1; ++ell) {
              slowing.set(mm_row+g, mm_col+gp, 
                          coefficients[m_row][m_col].scattering[j][ell] 
                          * mgxs.scatter[g][gp][j]);
            }
          }
        }
      }
    }
  }
  dealii::Vector<double> bx(size);
  dealii::Vector<double> modes_all(size);
  for (int m = 0; m < num_modes; ++m)
    for (int g = 0; g < num_groups; ++g)
      modes_all[m*num_groups+g] = this->modes[m][g];
  slowing.vmult(bx, modes_all);
  for (int i = 0; i < size; ++i) {
    const double diff = std::abs(a_minus_kb(i, last) + bx[i]);
    // std::cout << a_minus_kb(i, last) << ", " << bx[i] << "\n";
    // AssertThrow(diff < 1e-12, dealii::ExcMessage(
    //   std::to_string(a_minus_kb(i, last))+"!="+std::to_string(-bx[i])));
    a_minus_kb.set(i, last, -bx[i]);
  }
  // }
  // solve
  // dst = 1;
  // dst /= dst.l2_norm();
  // dst *= src.l2_norm();
  dst = 0;
  // solver.connect([](const unsigned int iteration,
  //                   const double check_value,
  //                   const dealii::Vector<double>&) {
  //   // std::cout << iteration << ": " << check_value << std::endl;
  //   return dealii::SolverControl::success;
  // });
  solver.solve(a_minus_kb, dst, src, preconditioner);
}

// void EnergyMgFiss::solve_fixed_k_(
//     dealii::Vector<double> &dst,
//     const dealii::Vector<double> &src,
//     const double k_eigenvalue,
//     const std::vector<std::vector<InnerProducts>> &coefficients) {
//   const int num_modes = this->modes.size();
//   const int num_groups = dst.size() / num_modes;
//   AssertDimension(dst.size(), num_groups*num_modes);
//   AssertDimension(dst.size(), src.size());
//   for (int m = 0; m < num_modes; ++m) {
//     for (int g = 0; g < num_groups; ++g) {
//       this->modes[m][g] = src[m*num_groups+g];
//     }
//   }
// }

void EnergyMgFiss::set_fixed_k_matrix(
    dealii::FullMatrix<double> &a_minus_kb, const double k_eigenvalue,
    const std::vector<std::vector<InnerProducts>> &coefficients) {
  const int num_modes = coefficients.size();
  AssertDimension(num_modes, coefficients[0].size());
  const int num_groups = a_minus_kb.m() / num_modes;
  for (int m_row = 0; m_row < num_modes; ++m_row) {
    int mm_row = m_row * num_groups;
    for (int m_col = 0; m_col < num_modes; ++m_col) {
      int mm_col = m_col * num_groups;
      for (int g = 0; g < num_groups; ++g) {
        a_minus_kb.add(mm_row+g, mm_col+g,
                       -k_eigenvalue * coefficients[m_row][m_col].streaming);
        for (int j = 0; j < mgxs.total[g].size(); ++j) {
          a_minus_kb.add(mm_row+g, mm_col+g, 
              -k_eigenvalue * coefficients[m_row][m_col].collision[j] 
              * mgxs.total[g][j]);
          for (int gp = 0; gp < num_groups; ++gp) {
            for (int ell = 0; ell < 1; ++ell) {
              a_minus_kb.add(mm_row+g, mm_col+gp, 
                             -k_eigenvalue
                             * coefficients[m_row][m_col].scattering[j][ell] 
                             * mgxs.scatter[g][gp][j]);
            }
            a_minus_kb.add(mm_row+g, mm_col+gp,
                           mgxs.chi[g][j] 
                           * coefficients[m_row][m_col].fission[j] 
                           * mgxs.nu_fission[gp][j]);
          }
        }
      }
    }
  }
}

void EnergyMgFiss::get_inner_products(
    const dealii::Vector<double> &modes_,
    std::vector<std::vector<InnerProducts>> &inner_products) {
  const int num_modes = inner_products.size();
  AssertDimension(num_modes, inner_products[0].size());
  const int num_groups = modes_.size() / num_modes;
  modes.resize(num_modes);
  for (int m = 0; m < num_modes; ++m)
    for (int g = 0; g < num_groups; ++g)
      modes[m][g] = modes_[m*num_groups+g];
  std::vector<double> _;
  for (int m = 0; m < num_modes; ++m)
    this->get_inner_products(inner_products[m], _, m, 0);
}

double EnergyMgFiss::inner_product(const dealii::Vector<double> &left,
                                   const dealii::Vector<double> &right) {
  double result = 0;
  const int num_groups = mgxs.total.size();
  const int num_modes = left.size() / num_groups;
  for (int g = 0; g < num_groups; ++g) {
    int g_rev = num_groups - 1 - g;
    double lower = mgxs.group_structure[g_rev];
    if (lower == 0)
      lower = 1e-5;
    double width = std::log(mgxs.group_structure[g_rev+1]
                            / lower);
    for (int m = 0; m < num_modes; ++m) {
      int mg = m * num_groups + g;
      result += left[mg] * right[mg] / width;
    }
  }
  return result;
}

}