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

void EnergyMgFull::step(dealii::BlockVector<double>&,
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
  modes.back().sadd(1 - omega, omega, solution);
  // matrix.vmult(solution, modes.back());
  // source -= solution;
  // modes.back() += source;
  // modes.back().print(std::cout);
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

double EnergyMgFull::enrich(const double factor) {
  modes.emplace_back(mgxs.total.size());
  return 0;
}

void EnergyMgFull::normalize() {
  // modes.back() /= modes.back().l2_norm();
  // double sum = 0;
  // for (int g = 0; g < modes.back().size(); ++g)
  //   sum += modes.back()[g];
  // if (sum < 0)
  //   modes.back() *= -1;
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
  dealii::Vector<double> mode_row = modes[m_row];
  for (int g = 0; g < mode_row.size(); ++g) {
    int g_rev = mode_row.size() - 1 - g;
    // double width = mgxs.group_structure[g+1] - mgxs.group_structure[g];
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
    // std::cout << "e ip b " << inner_products_b[i] << std::endl;
  }
  AssertDimension(modes.size(), inner_products_x.size());
  for (int m = m_col_start; m < modes.size(); ++m) {
    inner_products_x[m] = 0;
    for (int g = 0; g < modes.back().size(); ++g) {
      inner_products_x[m].streaming += mode_row[g] * modes[m][g];
      for (int j = 0; j < mgxs.total[g].size(); ++j) {
        inner_products_x[m].collision[j] += 
            mode_row[g] * mgxs.total[g][j] * modes[m][g];
        for (int gp = 0; gp < mgxs.scatter[g].size(); ++gp) {
          for (int ell = 0; ell < 1; ++ell) {
            inner_products_x[m].scattering[j][ell] +=
                mode_row[g] 
                * mgxs.scatter[g][gp][j]
                * modes[m][gp];
          }
        }
      }
    }
    // std::cout << "e ip x " << inner_products_x[m].streaming << std::endl;
    for (int j = 0; j < inner_products_x[m].collision.size(); ++j) {
      // std::cout << "e ip x " << inner_products_x[m].collision[j] << std::endl;
      // std::cout << "e ip x " << inner_products_x[m].scattering[j][0] << std::endl;
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
  if (modes.size() * num_groups > 1000) {
    for (int m = 0; m < modes.size(); ++m) {
      int mm = m * num_groups;
      for (int g = 0; g < num_groups; ++g) {
        solution_u[mm+g] = modes[m][g];
      }
    }
    dealii::SolverControl control(1000, source_u.l2_norm()*1e-4);
    dealii::SolverGMRES<dealii::Vector<double>> solver(control);
    try {
      solver.solve(matrix_u, solution_u, source_u, 
                  dealii::PreconditionIdentity());
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