#include "energy_mg_full.h"

namespace aether::pgd::sn {

EnergyMgFull::EnergyMgFull(const vector2<double> &cross_sections_total,
                           const vector4<double> &cross_sections_scatter,
                           const std::vector<dealii::Vector<double>> &sources)
  : cross_sections_total(cross_sections_total), 
    cross_sections_scatter(cross_sections_scatter),
    sources(sources),
    matrix(cross_sections_total.size()),
    source(cross_sections_total.size()) {
  Assert(cross_sections_total.size() > 0, dealii::ExcInvalidState());
  AssertDimension(cross_sections_total.size(), cross_sections_scatter.size());
  const int num_ell = cross_sections_scatter[0].size();
  for (int g = 1; g < cross_sections_scatter.size(); ++g)
    AssertDimension(cross_sections_scatter[g].size(), num_ell);
}

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
  set_matrix(coefficients_x.back());
  coefficients_x.pop_back();
  set_source(coefficients_b, coefficients_x);
  matrix.gauss_jordan();
  matrix.vmult(modes.back(), source);
}

void EnergyMgFull::enrich() {
  modes.emplace_back(cross_sections_total.size());
  modes.back() = 1;
}

void EnergyMgFull::set_matrix(InnerProducts coefficients_x) {
  matrix = 0;
  for (int g = 0; g < cross_sections_total.size(); ++g) {
    matrix[g][g] += coefficients_x.streaming;
    for (int j = 0; j < cross_sections_total[g].size(); ++j) {
      matrix[g][g] +=  coefficients_x.collision[j] * cross_sections_total[g][j];
      for (int gp = 0; gp < cross_sections_scatter[g].size(); ++gp) {
        for (int ell = 0; ell < cross_sections_scatter[g][gp][j].size(); ++ell) {
          matrix[g][gp] -= coefficients_x.scattering[j][ell]
                           * cross_sections_scatter[g][gp][j][ell];
        }
      }
    }
  }
}

void EnergyMgFull::set_source(std::vector<double> coefficients_b, 
                              std::vector<InnerProducts> coefficients_x) {
  source = 0;
  for (int i = 0; i < coefficients_b.size(); ++i)
    source.add(coefficients_b[i], sources[i]);
  AssertDimension(coefficients_x.size(), modes.size() - 1);
  for (int m = 0; m < coefficients_x.size(); ++m) {
    for (int g = 0; g < cross_sections_total.size(); ++g) {
      source[g] -= coefficients_x[m].streaming * modes[m][g];
      for (int j = 0; j < cross_sections_total[g].size(); ++j) {
        source[g] -= coefficients_x[m].collision[j] 
                      * cross_sections_total[g][j] 
                      * modes[m][g];
        for (int gp = 0; gp < cross_sections_scatter[g].size(); ++gp) {
          for (int ell = 0; ell < cross_sections_scatter[g][gp][j].size(); ++ell) {
            source[g] -= coefficients_x[m].scattering[j][ell]
                          * cross_sections_scatter[g][gp][j][ell]
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
      for (int j = 0; j < cross_sections_total[g].size(); ++j) {
        inner_products_x[m].collision[j] += 
            modes.back()[g] * cross_sections_total[g][j] * modes[m][g];
        for (int gp = 0; gp < cross_sections_scatter[g].size(); ++gp) {
          for (int ell = 0; ell < cross_sections_scatter[g][gp][j].size(); 
               ++ell) {
            inner_products_x[m].scattering[j][ell] +=
                modes.back()[g] 
                * cross_sections_scatter[g][gp][j][ell]
                * modes[m][g];
          }
        }
      }
    }
  }
}

}  // namespace aether::pgd::sn