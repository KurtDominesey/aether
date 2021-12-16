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
    std::vector<std::vector<InnerProducts>> &coefficients, const double tol,
    const std::string eps_type, int num_modes) {
  AssertDimension(coefficients.size(), modes.size());
  // int num_modes = modes.size();
  if (num_modes == -1)
    num_modes = modes.size();
  else
    num_modes = std::min(int(modes.size()), num_modes);
  const int num_groups = modes[0].size();
  const int size = num_modes * num_groups;
  // set matrices
  // dealii::FullMatrix<double> slowing(modes.size() * num_groups);
  // dealii::FullMatrix<double> fission(modes.size() * num_groups);
  // dealii::PETScWrappers::FullMatrix slowing(size, size);
  // dealii::PETScWrappers::FullMatrix fission(size, size);
  dealii::PETScWrappers::SparseMatrix slowing(size, size, size);
  // MatSetOption(slowing, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
  dealii::PETScWrappers::SparseMatrix fission(size, size, size);
  for (int m_row = 0; m_row < num_modes; ++m_row) {
    int mm_row = m_row * num_groups;
    for (int m_col = 0; m_col < num_modes; ++m_col) {
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
  const int num_blocks = num_modes;
  int num_subblocks = (modes.size()-1) % num_blocks;
  std::cout << "num_subblocks: " << num_subblocks << "\n";
  int last = size - num_groups;
  if (num_subblocks == 0) {
    dealii::LAPACKFullMatrix_<double> a(num_groups);
    for (int g = 0; g < num_groups; ++g)
      for (int gp = 0; gp < num_groups; ++gp)
        a(g, gp) = slowing(last+g, last+gp);
    if (num_modes == 1)
      gs_lu.blocks.clear();
    gs_lu.initialize(a);
    std::cout << "init'd growing LU\n";
  } else {
    int m0 = num_modes - 1 - num_subblocks;
    dealii::LAPACKFullMatrix_<double> b, c, d;
    b.reinit(num_subblocks*num_groups, num_groups);
    c.reinit(num_groups, num_subblocks*num_groups);
    d.reinit(num_groups);
    std::cout << "num modes: " << num_modes 
              << ", m0: " << m0
              << ", last:" << last << "\n";
    for (int m = m0; m < num_modes; ++m) {
      int mm = m * num_groups;
      int mo = (m - m0) * num_groups;
      for (int g = 0; g < num_groups; ++g) {
        for (int gp = 0; gp < num_groups; ++gp) {
          if (m < num_modes - 1) {
            b(mo+g, gp) = slowing(mm+g, last+gp);  // last block column
            c(g, mo+gp) = slowing(last+g, mm+gp);  // last block row
          } else {
            d(g, gp) = slowing(last+g, last+gp);  // last diagonal block
          }
        }
      }
    }
    std::cout << "growing LU\n";
    gs_lu.grow(b, c, d);
    std::cout << "grew LU\n";
  }
  gs_lu.matrix = &slowing;
  PETScWrappers::MatrixFreeWrapper<PreconditionBlockGrowingLU<
        dealii::PETScWrappers::SparseMatrix, double>> lu_petsc(
      MPI_COMM_WORLD, size, size, size, size, gs_lu);
  // set solution vector
  // dealii::Vector<double> solution(modes.size() * num_groups);
  std::vector<dealii::PETScWrappers::MPI::Vector> eigenvectors;
  eigenvectors.emplace_back(MPI_COMM_WORLD, size, size);
  eigenvectors[0].compress(dealii::VectorOperation::add);
  for (int m = 0; m < num_modes; ++m) {
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
  rayleigh = this->eigenvalue;
  // return rayleigh; //!!!
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
  int max_iters = 10;
  if (eps_type == "jd")
    max_iters = 20;
  dealii::IterationNumberControl control(max_iters, tol);
  control.enable_history_data();
  control.log_history(true);
  std::unique_ptr<dealii::SLEPcWrappers::SolverBase> eigensolver_ptr;
  if (eps_type == "krylovschur")
    eigensolver_ptr = 
        std::make_unique<dealii::SLEPcWrappers::SolverKrylovSchur>(control);
  else if (eps_type == "jd")
    eigensolver_ptr = 
        std::make_unique<dealii::SLEPcWrappers::SolverJacobiDavidson>(control);
  else
    AssertThrow(false, dealii::ExcNotImplemented());
  dealii::SLEPcWrappers::SolverBase &eigensolver = *eigensolver_ptr;
  // dealii::SLEPcWrappers::SolverKrylovSchur eigensolver(control);
  // dealii::SLEPcWrappers::SolverGeneralizedDavidson eigensolver(control);
  // dealii::SLEPcWrappers::SolverJacobiDavidson eigensolver(control);
  eigensolver.set_target_eigenvalue(rayleigh/*+1e-2*/);
  eigensolver.set_which_eigenpairs(EPS_LARGEST_MAGNITUDE);
  if (eigenvectors[0].l2_norm() > 0)
    eigensolver.set_initial_space(eigenvectors);
  // set preconditioner
  bool use_pc = false;
  // if (use_pc) {
    // dealii::PETScWrappers::FullMatrix shifted(size, size);
    // shifted.add(1, fission);
    // shifted.add(-rayleigh, slowing);
    unsigned int sum = 0;
    unsigned int sum_a = 0;
    std::vector<unsigned int> row_sizes(size);
    for (int r = 0; r < size; ++r) {
      for (int c = 0; c < size; ++c) {
        if (slowing.el(r, c) != 0 || r == c) {
          row_sizes[r] += 1;
          sum++;
        } else if (fission.el(r, c) != 0) {
          sum_a++;
        }
      }
    }
    std::cout << "sparsity B: " << (double(sum)/double(size*size)) << "\n";
    std::cout << "sparsity A+B: " 
              << (double(sum+sum_a)/double(size*size)) << "\n";
    dealii::DynamicSparsityPattern dsp(size);
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        if (slowing.el(i, j) != 0) {
          dsp.add(i, j);
        }
      }
    }
    dealii::SparsityPattern sp;
    sp.copy_from(dsp);
    // dealii::PETScWrappers::FullMatrix shifted(size, size);
    // dealii::PETScWrappers::SparseMatrix shifted(size, size, size);
    dealii::PETScWrappers::SparseMatrix shifted(sp);
    // dealii::FullMatrix<double> shifted_full(size);
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        // if (slowing.el(i, j) == 0 && i != j);
        //   continue;
        if (slowing.el(i, j) != 0) {
          // std::cout << "i, j: " << i << ", " << j << "\n";
          shifted.add(i, j, /*fission.el(i, j) - rayleigh */ slowing.el(i, j));
          // shifted_full(i, j) = fission.el(i, j) - rayleigh * slowing.el(i, j);
        }
      }
    }
    shifted.compress(dealii::VectorOperation::add);
    for (int i = 0; i < size; ++i) {
      // std::cout << shifted.el(i, i) << " / " << slowing.el(i, i) << "\n";
      // AssertThrow(shifted.el(i, i) != 0, dealii::ExcMessage(std::to_string(i)));
    }
    std::cout << "shifted it\n";
    if (eps_type == "jd") {
    // dealii::PETScWrappers::PreconditionNone preconditioner(shifted);
    dealii::PETScWrappers::PreconditionSOR preconditioner(shifted);
    // try dealii operators
    // dealii::SparsityPattern pattern;
    // pattern.copy_from(shifted_full);
    // dealii::SparseMatrix<double> shifted_sparse(pattern);
    // shifted_sparse.copy_from(shifted_full);
    // shifted_full = 0;
    // using BlockSOR = dealii::PreconditionBlockSOR<dealii::SparseMatrix<double>>;
    // BlockSOR shifted_gs;
    // shifted_gs.initialize(shifted_sparse, BlockSOR::AdditionalData(num_groups));
    // aether::PETScWrappers::MatrixFreeWrapper<BlockSOR> shifted_gs_mf(
    //     MPI_COMM_WORLD, size, size, size, size, shifted_gs);
    // aether::PETScWrappers::PreconditionerShell shifted_gs_pc(shifted_gs_mf);
    // dealii::SolverControl control_dummy(1, 0);
    // dealii::PETScWrappers::SolverPreOnly solver_pc(control_dummy);
    dealii::IterationNumberControl control_pc(50, 0);
    dealii::PETScWrappers::SolverGMRES solver_pc(control_pc);
    // solver_pc.initialize(shifted_gs_pc);
    solver_pc.initialize(preconditioner);
    // aether::SLEPcWrappers::TransformationPreconditioner stprecond(
    //     MPI_COMM_WORLD, shifted_gs_mf);
    aether::SLEPcWrappers::TransformationPreconditioner stprecond(
        MPI_COMM_WORLD, shifted);
    stprecond.set_matrix_mode(ST_MATMODE_SHELL);
    stprecond.set_solver(solver_pc);
      eigensolver.set_transformation(stprecond);
    // }
  }
  bool do_sinvert = false;
  // if (do_sinvert) {
    dealii::SLEPcWrappers::TransformationShift shift_invert(
        MPI_COMM_WORLD,
        dealii::SLEPcWrappers::TransformationShift::AdditionalData(
          rayleigh /*1.32665514562608*/));
    // dealii::SLEPcWrappers::TransformationShift shift_invert(MPI_COMM_WORLD);
    // shift_invert.set_matrix_mode(ST_MATMODE_SHELL);
    dealii::IterationNumberControl control_inv(5, 0);
    // dealii::PETScWrappers::SparseDirectMUMPS solver_inv(control_inv);
    // dealii::PETScWrappers::SolverGMRES solver_inv(control_inv,
    //                                               MPI_COMM_WORLD);
    // dealii::PETScWrappers::SolverGMRES solver_inv(control_inv, MPI_COMM_WORLD,
    //     dealii::PETScWrappers::SolverGMRES::AdditionalData(30));
    // aether::PETScWrappers::SolverFGMRES solver_inv(
    //     control_inv, MPI_COMM_WORLD, 
    //     aether::PETScWrappers::SolverFGMRES::AdditionalData(10));
    // dealii::PETScWrappers::PreconditionNone identity(slowing);
    // dealii::PETScWrappers::PreconditionSOR slowing_gs(slowing);
    // dealii::PETScWrappers::PreconditionParaSails slowing_pc(slowing,
    //     dealii::PETScWrappers::PreconditionParaSails::AdditionalData(0));
    // dealii::PETScWrappers::PreconditionILU slowing_pc(shifted);
    // dealii::PETScWrappers::PreconditionParaSails slowing_pc(shifted,
    //     dealii::PETScWrappers::PreconditionParaSails::AdditionalData(0));
    // solver_inv.initialize(slowing_gs);
    // shift_invert.set_solver(solver_inv);
    PETScWrappers::PreconditionerShell pc_mat(lu_petsc);
    dealii::PETScWrappers::SolverPreOnly solver_inv(control_inv);
    solver_inv.initialize(pc_mat);
    shift_invert.set_solver(solver_inv);
    shift_invert.set_matrix_mode(ST_MATMODE_SHELL);
    if (eps_type == "krylovschur")
      eigensolver.set_transformation(shift_invert);
  // }
  // solve eigenproblem
  std::vector<double> eigenvalues;
  try {
    std::cout << "solving!\n";
    eigensolver.solve(fission, slowing, eigenvalues, eigenvectors);
    if (eigenvalues[0] < 0) {
      eigenvalues[0] *= -1;
      eigenvectors[0] *= -1;
    }
    bool do_adjoint = false;
    std::vector<dealii::PETScWrappers::MPI::Vector> eigenvectors_adj;
    if (do_adjoint) {  // do adjoint
      fission.transpose();
      slowing.transpose();
      std::vector<double> eigenvalues_adj(eigenvalues);
      eigenvectors_adj = eigenvectors;
      if (preconditioner_adj_ptr.get() == NULL) {
        shifted.transpose();
        preconditioner_adj_ptr = 
          std::make_unique<dealii::PETScWrappers::PreconditionILU>(shifted);
      }
      solver_inv.initialize(*preconditioner_adj_ptr);
      shift_invert.set_solver(solver_inv);
      eigensolver.set_transformation(shift_invert);
      eigensolver.solve(fission, slowing, eigenvalues_adj, eigenvectors_adj);
      std::cout << "FORWARD EIGENVALUE: " << eigenvalues[0] << "\n"
                << "ADJOINT EIGENVALUE: " << eigenvalues_adj[0] << "\n"
                << "DIFFERENCE [pcm]: " 
                << (1e5 * (eigenvalues_adj[0]-eigenvalues[0])) << "\n";
      eigenvectors_adj[0] *= -1;
      // auto bx(eigenvectors[0]);
      // auto bx_adj(eigenvectors_adj[0]);
      // fission.vmult(bx, eigenvectors[0]);
      // fission.vmult(bx_adj, eigenvectors_adj[0]);
      // std::cout << "biorthogonal? "
      //           // << (eigenvectors[0]*bx_adj) << " "
      //           << (eigenvectors_adj[0]*bx) << "\n";
      if (test_funcs.empty())
        test_funcs.resize(num_modes, dealii::Vector<double>(num_groups));
    }
    for (int m = 0, i = 0; m < num_modes; ++m) {
      for (int g = 0; g < num_groups; ++g, ++i) {
        modes[m][g] = eigenvectors[0][i];
        if (do_adjoint) {
          test_funcs[m][g] = eigenvectors_adj[0][i];
        }
      }
    }
    std::cout << "k-updatestep: " << eigenvalues[0] << "\n";
    this->eigenvalue = eigenvalues[0];
    const std::vector<double> &history = control.get_history_data();
    for (int i = 0; i < history.size(); ++i)
      std::cout << history[i] << "\n";
    return eigenvalues[0];
  } catch (...) {
    std::cout << "failed\n";
    for (int m = 0, i = 0; m < num_modes; ++m) {
      for (int g = 0; g < num_groups; ++g, ++i) {
        AssertThrow(modes[m][g] == eigenvectors[0][i],
                    dealii::ExcInvalidState())
        modes[m][g] = eigenvectors[0][i];
      }
    }
    const std::vector<double> &history = control.get_history_data();
    for (int i = 0; i < history.size(); ++i)
      std::cout << history[i] << "\n";
    return rayleigh;  // !
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
  const int num_groups = mgxs.total.size();
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
  for (int m = 0; m < num_modes; ++m) {
    for (int g = 0; g < num_groups; ++g) {
      int g_rev = num_groups - 1 - g;
      double lower = mgxs.group_structure[g_rev];
      if (lower == 0)
        lower = 1e-5;
      double width = std::log(mgxs.group_structure[g_rev+1]
                              / lower);
      a_minus_kb.set(last, m*num_groups+g, 1e2*2*modes[m][g]/*width */);
    }
  }
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
  const int num_groups = mgxs.total.size();
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