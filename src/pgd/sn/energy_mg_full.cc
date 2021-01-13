#include "energy_mg_full.h"

namespace aether::pgd::sn {

EnergyMgFull::EnergyMgFull(const Mgxs &mgxs,
                           const std::vector<dealii::Vector<double>> &sources)
  : mgxs(mgxs),
    sources(sources),
    matrix(mgxs.total.size()),
    source(mgxs.total.size()) {}

void EnergyMgFull::vmult(dealii::BlockVector<double>&,
                         const dealii::BlockVector<double>&,
                         std::vector<InnerProducts> coefficients_x,
                         std::vector<double> coefficients_b) {
  throw dealii::ExcNotImplemented();
  set_matrix(coefficients_x.back());
  coefficients_x.pop_back();
  set_source(coefficients_b, coefficients_x);
  matrix.vmult(modes.back(), source);
}

double EnergyMgFull::step(dealii::Vector<double> &delta,
                        const dealii::BlockVector<double>&,
                        std::vector<InnerProducts> coefficients_x,
                        std::vector<double> coefficients_b,
                        double omega) {
  AssertDimension(coefficients_x.size(), modes.size());
  AssertDimension(coefficients_b.size(), sources.size());
  set_matrix(coefficients_x.back());
  coefficients_x.pop_back();
  set_source(coefficients_b, coefficients_x);
  // matrix.print(std::cout);
  // source.print(std::cout);
  matrix.gauss_jordan();
  dealii::Vector<double> solution(modes.back());
  matrix.vmult(solution, source);
  ////
  // dealii::FullMatrix<double> matrix_n(matrix.m()+1, matrix.n()+1);
  // for (int i = 0; i < matrix.m(); ++i)
  //   for (int j = 0; j < matrix.n(); ++j)
  //     matrix_n[i][j] = matrix[i][j];
  // for (int j = 0; j < matrix.n(); ++j)
  //   matrix_n[matrix.m()][j] = 1;
  // dealii::Vector<double> source_n(source);
  // source_n.grow_or_shrink(source.size()+1);
  // solution.grow_or_shrink(source.size()+1);
  // source_n[source.size()] = 1;
  // matrix_n.gauss_jordan();
  // matrix_n.vmult(solution, source_n);
  // solution.grow_or_shrink(source.size());
  ////
  delta = solution;
  delta -= modes.back();
  return modes.back().l2_norm();
  // modes.back().sadd(1 - omega, omega, solution);
  // matrix.vmult(solution, modes.back());
  // source -= solution;
  // modes.back() += source;
  // modes.back().print(std::cout);
}

void EnergyMgFull::take_step(const double factor, 
                             const dealii::Vector<double> &delta) {
  modes.back().add(factor, delta);
}

double EnergyMgFull::get_residual(
                        std::vector<InnerProducts> coefficients_x,
                        std::vector<double> coefficients_b) {
  AssertDimension(coefficients_x.size(), modes.size());
  AssertDimension(coefficients_b.size(), sources.size());
  set_matrix(coefficients_x.back());
  coefficients_x.pop_back();
  set_source(coefficients_b, coefficients_x);
  dealii::Vector<double> operated(modes.back());
  matrix.vmult(operated, modes.back());
  dealii::Vector<double> residual(source);
  residual -= operated;
  return residual.l2_norm() / source.l2_norm();
}

double EnergyMgFull::line_search(
    std::vector<double> &c,
    const dealii::Vector<double> &step,
    const InnerProducts &coefficients_x_mode_step,
    const InnerProducts &coefficients_x_step_step,
    std::vector<InnerProducts> coefficients_x_mode_mode,
    std::vector<InnerProducts> coefficients_x_step_mode,
    const std::vector<double> &coefficients_b_mode,
    const std::vector<double> &coefficients_b_step) {
  AssertDimension(coefficients_x_mode_mode.size(), modes.size());
  AssertDimension(coefficients_x_step_mode.size(), modes.size());
  AssertDimension(coefficients_b_mode.size(), sources.size());
  AssertDimension(coefficients_b_step.size(), sources.size());
  set_matrix(coefficients_x_mode_mode.back());
  coefficients_x_mode_mode.pop_back();
  dealii::FullMatrix<double> a00(matrix);
  set_matrix(coefficients_x_step_mode.back());
  coefficients_x_step_mode.pop_back();
  dealii::FullMatrix<double> a10(matrix);
  set_matrix(coefficients_x_mode_step);
  dealii::FullMatrix<double> a01(matrix);
  set_matrix(coefficients_x_step_step);
  dealii::FullMatrix<double> a11(matrix);
  set_source(coefficients_b_mode, coefficients_x_mode_mode);
  dealii::Vector<double> f0(source);
  set_source(coefficients_b_step, coefficients_x_step_mode);
  dealii::Vector<double> f1(source);
  dealii::Vector<double> v0(f0);
  v0 *= -1;
  a00.vmult_add(v0, modes.back());
  dealii::Vector<double> v1(f1);
  v1 *= -1;
  a10.vmult_add(v1, modes.back());
  a01.vmult_add(v1, modes.back());
  a00.vmult_add(v1, step);
  dealii::Vector<double> v2(v0.size());
  a11.vmult_add(v2, modes.back());
  a10.vmult_add(v2, step);
  a01.vmult_add(v2, step);
  dealii::Vector<double> v3(v0.size());
  a11.vmult_add(v3, step);
  // invert ?
  dealii::FullMatrix<double> a(a00);
  a.add(1, a01);
  a.add(1, a10);
  a.add(1, a11);
  a.gauss_jordan();
  dealii::Vector<double> v(v0.size());
  a.vmult(v, v0);
  v0 = v;
  a.vmult(v, v1);
  v1 = v;
  a.vmult(v, v2);
  v2 = v;
  a.vmult(v, v3);
  v3 = v;
  // get coefficients
  // std::vector<double> c(7);
  double norm = modes.back().l2_norm();
  c[0] += v0 * v0 / norm;
  c[1] += 2 * (v1 * v0) / norm;
  c[2] += (2 * (v2 * v0) + v1 * v1) / norm;
  c[3] += 2 * (v2 * v1) / norm;
  c[4] += (2 * (v3 * v1) + v2 * v2) / norm;
  c[5] += 2 * (v3 * v2) / norm;
  c[6] += v3 * v3 / norm;
  return 1; // !!
  // std::function<double(std::vector<double>, double)> 
  auto res = [](std::vector<double> c, std::complex<double> a) {
    std::complex<double> r = c[0];
    for (int i = 1; i < c.size(); ++i)
      r += c[i] * std::pow(a, i);
    return r;
  };
  f1 += f0;
  std::cout << std::sqrt(std::abs(res(c, 0.)))/f0.l2_norm() << " --> " 
            << std::sqrt(std::abs(res(c, 1.)))/f1.l2_norm() << std::endl;
  // return 1;
  std::vector<double> dc(c.size()-1);
  for (int i = 0; i < dc.size(); ++i)
    dc[i] = (i+1) * c[i+1];
  std::vector<double> ddc(dc.size()-1);
  for (int i = 0; i < ddc.size(); ++i)
    ddc[i] = (i+1) * dc[i+1];
  std::vector<std::complex<double>> z(dc.size());
  for (int i = 0; i < z.size(); ++i)
    z[i] = std::pow(std::complex<double>(0.4, 0.9), i);
  auto p = std::bind(res, dc, std::placeholders::_1);
  auto pp = std::bind(res, ddc, std::placeholders::_1);
  auto aberth = [p, pp](std::vector<std::complex<double>> &z) {
    std::complex<double> w;
    for (int n = 0; n < 50; ++n) {
      for (int i = 0; i < z.size(); ++i) {
        std::complex<double> ratio = p(z[i]) / pp(z[i]);
        std::complex<double> sum = 0;
        for (int j = 0; j < z.size(); ++j)
          if (i != j)
            sum += 1. / (z[i] - z[j]);
        w = ratio / (1. - ratio * sum);
        if (std::isfinite(std::abs(w)))
          z[i] -= w;
      }
    }
  };
  aberth(z);
  double lambda = 1;
  double res_lambda = std::numeric_limits<double>::infinity(); //std::abs(res(c, lambda));
  for (int i = 0; i < z.size(); ++i) {
    // std::cout << z[i] << " ";
    if (z[i].imag() < 1e-16 && std::abs(z[i].real()) > 5e-2) {
      double res_zi = std::abs(res(c, z[i].real()));
      if (res_zi < res_lambda) {
        lambda = z[i].real();
        res_lambda = res_zi;
        std::cout << lambda << " ";
      }
    }
  }
  std::cout << std::endl;
  // std::cout << v0.l2_norm() / f0.l2_norm() << std::endl;
  return lambda;
}

double EnergyMgFull::enrich(const double factor) {
  modes.emplace_back(mgxs.total.size());
  return 0;
}

double EnergyMgFull::normalize() {
  return 0;
  // return modes.back().l2_norm();
  // modes.back() /= modes.back().l2_norm();
  // double sum = 0;
  // for (int g = 0; g < modes.back().size(); ++g)
  //   sum += modes.back()[g];
  // if (sum < 0)
  //   modes.back() *= -1;
}

void EnergyMgFull::scale(double factor) {
  modes.back() *= factor;
}

void EnergyMgFull::set_matrix(InnerProducts coefficients_x) {
  matrix = 0;
  // std::cout << coefficients_x.streaming << std::endl;
  for (int g = 0; g < mgxs.total.size(); ++g) {
    matrix[g][g] += coefficients_x.streaming;
    for (int j = 0; j < mgxs.total[g].size(); ++j) {
      matrix[g][g] +=  coefficients_x.collision[j] * mgxs.total[g][j];
      for (int gp = 0; gp < mgxs.scatter[g].size(); ++gp) {
        for (int ell = 0; ell < 1; ++ell) {
          matrix[g][gp] += coefficients_x.scattering[j][ell]
                           * mgxs.scatter[g][gp][j];
        }
      }
    }
  }
}

void EnergyMgFull::set_source(std::vector<double> coefficients_b, 
                              std::vector<InnerProducts> coefficients_x) {
  source = 0;
  AssertDimension(coefficients_b.size(), sources.size());
  for (int i = 0; i < coefficients_b.size(); ++i) {
    source.add(coefficients_b[i], sources[i]);
  }
  AssertDimension(coefficients_x.size(), modes.size() - 1);
  for (int m = 0; m < coefficients_x.size(); ++m) {
    for (int g = 0; g < mgxs.total.size(); ++g) {
      source[g] -= coefficients_x[m].streaming * modes[m][g];
      for (int j = 0; j < mgxs.total[g].size(); ++j) {
        source[g] -= coefficients_x[m].collision[j] 
                      * mgxs.total[g][j] 
                      * modes[m][g];
        for (int gp = 0; gp < mgxs.scatter[g].size(); ++gp) {
          for (int ell = 0; ell < 1; ++ell) {
            source[g] -= coefficients_x[m].scattering[j][ell]
                          * mgxs.scatter[g][gp][j]
                          * modes[m][gp];
          }
        }
      }
    }
  }
}

void EnergyMgFull::get_inner_products(
    std::vector<InnerProducts> &inner_products_x,
    std::vector<double> &inner_products_b) {
  const int m_row = modes.size() - 1;
  const int m_col_start = 0;
  get_inner_products(inner_products_x, inner_products_b, m_row, m_col_start);
}

void EnergyMgFull::get_inner_products(
    std::vector<InnerProducts> &inner_products_x,
    std::vector<double> &inner_products_b,
    const int m_row, const int m_col_start) {
  dealii::Vector<double> &left = modes[m_row];
  get_inner_products_b(inner_products_b, left);
  AssertDimension(modes.size(), inner_products_x.size());
  for (int m = m_col_start; m < modes.size(); ++m) {
    dealii::Vector<double> &right = modes[m];
    get_inner_products_x(inner_products_x[m], left, right);
  }
}

void EnergyMgFull::get_inner_products_x(
    std::vector<InnerProducts> &inner_products_x,
    const dealii::Vector<double> &left) {
  AssertDimension(modes.size(), inner_products_x.size());
  for (int m = 0; m < modes.size(); ++m) {
    dealii::Vector<double> &right = modes[m];
    get_inner_products_x(inner_products_x[m], left, right);
  }
}

void EnergyMgFull::get_inner_products_x(
    InnerProducts &inner_products_x,
    const dealii::Vector<double> &right) {
  get_inner_products_x(inner_products_x, modes.back(), right);
}

void EnergyMgFull::get_inner_products_b(
    std::vector<double> &inner_products_b,
    const dealii::Vector<double> &left) {
  dealii::Vector<double> mode_row(left);
  for (int g = 0; g < mode_row.size(); ++g) {
    int g_rev = mode_row.size() - 1 - g;
    double width = std::log(mgxs.group_structure[g+1]/mgxs.group_structure[g]);
    AssertThrow(width > 0, dealii::ExcDivideByZero());
    mode_row[g_rev] /= width;
  }
  AssertDimension(sources.size(), inner_products_b.size());
  for (int i = 0; i < sources.size(); ++i) {
    inner_products_b[i] = 0;
    for (int g = 0; g < modes.back().size(); ++g) {
      inner_products_b[i] += mode_row[g] * sources[i][g];
    }
  }
}
  
void EnergyMgFull::get_inner_products_x(
    InnerProducts &inner_products_x,
    const dealii::Vector<double> &left,
    const dealii::Vector<double> &right) {
  dealii::Vector<double> mode_row(left);
  for (int g = 0; g < mode_row.size(); ++g) {
    int g_rev = mode_row.size() - 1 - g;
    double width = std::log(mgxs.group_structure[g+1]/mgxs.group_structure[g]);
    AssertThrow(width > 0, dealii::ExcDivideByZero());
    mode_row[g_rev] /= width;
  }
  inner_products_x = 0;
  for (int g = 0; g < modes.back().size(); ++g) {
    inner_products_x.streaming += mode_row[g] * right[g];
    for (int j = 0; j < mgxs.total[g].size(); ++j) {
      inner_products_x.collision[j] += 
          mode_row[g] * mgxs.total[g][j] * right[g];
      for (int gp = 0; gp < mgxs.scatter[g].size(); ++gp) {
        inner_products_x.fission[j] += mode_row[g] 
                                       * mgxs.chi[g][j]
                                       * mgxs.nu_fission[gp][j]
                                       * right[gp];
        for (int ell = 0; ell < 1; ++ell) {
          inner_products_x.scattering[j][ell] +=
              mode_row[g] 
              * mgxs.scatter[g][gp][j]
              * right[gp];
        }
      }
    }
  }
}

void EnergyMgFull::update(
      std::vector<std::vector<InnerProducts>> coefficients_x,
      std::vector<std::vector<double>> coefficients_b) {
  AssertDimension(coefficients_x.size(), modes.size());
  AssertDimension(coefficients_b.size(), modes.size());
  const int num_groups = modes[0].size();
  dealii::FullMatrix<double> matrix_u(modes.size() * num_groups);
  dealii::Vector<double> source_u(modes.size() * num_groups);
  for (int m_row = 0; m_row < modes.size(); ++m_row) {
    int mm_row = m_row * num_groups;
    // Set matrix
    for (int m_col = 0; m_col < modes.size(); ++m_col) {
      int mm_col = m_col * num_groups;
      for (int g = 0; g < mgxs.total.size(); ++g) {
        matrix_u[mm_row+g][mm_col+g] += coefficients_x[m_row][m_col].streaming;
        for (int j = 0; j < mgxs.total[g].size(); ++j) {
          matrix_u[mm_row+g][mm_col+g] += 
              coefficients_x[m_row][m_col].collision[j] * mgxs.total[g][j];
          for (int gp = 0; gp < mgxs.scatter[g].size(); ++gp) {
            for (int ell = 0; ell < 1; ++ell) {
              matrix_u[mm_row+g][mm_col+gp] += 
                  coefficients_x[m_row][m_col].scattering[j][ell]
                  * mgxs.scatter[g][gp][j];
            }
          }
        }
      }
    }
    // Set source
    for (int g = 0; g < num_groups; ++g) {
      for (int i = 0; i < coefficients_b[m_row].size(); ++i) {
        source_u[mm_row+g] += coefficients_b[m_row][i] * sources[i][g];
      }
    }
  }
  dealii::Vector<double> solution_u(source_u.size());
  if (true) {
    for (int m = 0; m < modes.size(); ++m) {
      int mm = m * num_groups;
      for (int g = 0; g < num_groups; ++g) {
        solution_u[mm+g] = modes[m][g];
      }
    }
    dealii::SolverControl control(500, 1e-6);
    dealii::SolverGMRES<dealii::Vector<double>> solver(control,
       dealii::SolverGMRES<dealii::Vector<double>>::AdditionalData(52));
    solver.connect([](const unsigned int iteration,
                      const double check_value,
                      const dealii::Vector<double>&) {
      std::cout << "update " << iteration << ": " << check_value << std::endl;
      return dealii::SolverControl::success;
    });
    dealii::SparsityPattern pattern;
    pattern.copy_from(matrix_u);
    dealii::SparseMatrix<double> matrix_u_sp(pattern);
    matrix_u_sp.copy_from(matrix_u);
    // dealii::PreconditionSOR<dealii::SparseMatrix<double>> preconditioner;
    // preconditioner.initialize(matrix_u_sp);
    dealii::PreconditionBlockSOR<dealii::SparseMatrix<double>> preconditioner;
    preconditioner.initialize(matrix_u_sp,
        dealii::PreconditionBlock<dealii::SparseMatrix<double>>::AdditionalData(
          num_groups));
    // dealii::PreconditionIdentity preconditioner;
    try {
      solver.solve(matrix_u_sp, solution_u, source_u, preconditioner);
    } catch (dealii::SolverControl::NoConvergence &failure) {
      std::cout << "failure in updating energy\n";
      failure.print_info(std::cout);
    }
  } else {
    matrix_u.gauss_jordan();
    matrix_u.vmult(solution_u, source_u);
  }
  for (int m = 0; m < modes.size(); ++m) {
    int mm = m * num_groups;
    for (int g = 0; g < num_groups; ++g) {
      modes[m][g] = solution_u[mm+g];
    }
  }
}

}  // namespace aether::pgd::sn