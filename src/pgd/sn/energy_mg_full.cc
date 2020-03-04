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
  matrix.print(std::cout);
  source.print(std::cout);
  matrix.gauss_jordan();
  dealii::Vector<double> solution(modes.back());
  matrix.vmult(solution, source);
  modes.back().sadd(1 - omega, omega, solution);
  modes.back().print(std::cout);
}

void EnergyMgFull::enrich() {
  modes.emplace_back(mgxs.total.size());
  modes.back() = 1;
  normalize();
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
  AssertDimension(sources.size(), inner_products_b.size());
  for (int i = 0; i < sources.size(); ++i) {
    inner_products_b[i] = 0;
    for (int g = 0; g < modes.back().size(); ++g) {
      inner_products_b[i] += modes.back()[g] * sources[i][g];
    }
    std::cout << "e ip b " << inner_products_b[i] << std::endl;
  }
  AssertDimension(modes.size(), inner_products_x.size());
  for (int m = 0; m < modes.size(); ++m) {
    inner_products_x[m] = 0;
    for (int g = 0; g < modes.back().size(); ++g) {
      inner_products_x[m].streaming += modes.back()[g] * modes[m][g];
      for (int j = 0; j < mgxs.total[g].size(); ++j) {
        inner_products_x[m].collision[j] += 
            modes.back()[g] * mgxs.total[g][j] * modes[m][g];
        for (int gp = 0; gp < mgxs.scatter[g].size(); ++gp) {
          for (int ell = 0; ell < 1; ++ell) {
            inner_products_x[m].scattering[j][ell] +=
                modes.back()[g] 
                * mgxs.scatter[g][gp][j]
                * modes[m][gp];
          }
        }
      }
    }
    std::cout << "e ip x " << inner_products_x[m].streaming << std::endl;
    for (int j = 0; j < inner_products_x[m].collision.size(); ++j) {
      std::cout << "e ip x " << inner_products_x[m].collision[j] << std::endl;
      std::cout << "e ip x " << inner_products_x[m].scattering[j][0] << std::endl;
    }
  }
}

}  // namespace aether::pgd::sn