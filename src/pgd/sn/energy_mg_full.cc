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
                        std::vector<double> coefficients_b) {
  AssertDimension(coefficients_x.size(), modes.size());
  AssertDimension(coefficients_b.size(), sources.size());
  set_matrix(coefficients_x.back());
  coefficients_x.pop_back();
  set_source(coefficients_b, coefficients_x);
  matrix.print(std::cout);
  source.print(std::cout);
  matrix.gauss_jordan();
  matrix.vmult(modes.back(), source);
}

void EnergyMgFull::enrich() {
  modes.emplace_back(mgxs.total.size());
  modes.back() = 1;
}

void EnergyMgFull::normalize() {
  modes.back() /= modes.back().l2_norm();
}

void EnergyMgFull::set_matrix(InnerProducts coefficients_x) {
  matrix = 0;
  for (int g = 0; g < mgxs.total.size(); ++g) {
    std::cout << coefficients_x.streaming << std::endl;
    matrix[g][g] += coefficients_x.streaming;
    for (int j = 0; j < mgxs.total[g].size(); ++j) {
      matrix[g][g] +=  coefficients_x.collision[j] * mgxs.total[g][j];
      for (int gp = 0; gp < mgxs.scatter[g].size(); ++gp) {
        for (int ell = 0; ell < 1; ++ell) {
          matrix[g][gp] -= coefficients_x.scattering[j][ell]
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
    std::cout << coefficients_b[i] << std::endl;
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
                * modes[m][g];
          }
        }
      }
    }
  }
}

}  // namespace aether::pgd::sn