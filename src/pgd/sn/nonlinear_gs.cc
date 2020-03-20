#include "pgd/sn/nonlinear_gs.h"

namespace aether::pgd::sn {

NonlinearGS::NonlinearGS(std::vector<LinearInterface*> &linear_ops,
                         int num_materials, int num_legendre, int num_sources)
    : linear_ops(linear_ops),
      inner_products_x(linear_ops.size()),
      inner_products_b(linear_ops.size(), std::vector<double>(num_sources)),
      inner_products_all_x(linear_ops.size()),
      inner_products_all_b(linear_ops.size()),
      inner_products_one(num_materials, num_legendre) {
  inner_products_one = 1;
}

double NonlinearGS::step(dealii::BlockVector<double> x, 
                       const dealii::BlockVector<double> b,
                       const bool should_normalize) {
  double res_max = 0;
  std::vector<InnerProducts> coefficients_x(inner_products_x[0].size(),
                                            inner_products_x[0].back());
  std::vector<double> coefficients_b(inner_products_b[0].size());
  for (int i = 0; i < linear_ops.size(); ++i) {
    for (int m = 0; m < coefficients_x.size(); ++m)
      coefficients_x[m] = 1;
    for (int n = 0; n < coefficients_b.size(); ++n)
      coefficients_b[n] = 1;
    for (int j = 0; j < linear_ops.size(); ++j) {
      if (i == j)
        continue;
      for (int m = 0; m < coefficients_x.size(); ++m)
        coefficients_x[m] *= inner_products_x[j][m];
      for (int n = 0; n < coefficients_b.size(); ++n)
        coefficients_b[n] *= inner_products_b[j][n];
    }
    double res = linear_ops[i]->get_residual(coefficients_x, coefficients_b);
    res_max = std::max(res, res_max);
    linear_ops[i]->step(x, b, coefficients_x, coefficients_b);
    if (should_normalize)
      linear_ops[i]->normalize();
    linear_ops[i]->get_inner_products(inner_products_x[i], inner_products_b[i]);
  }
  return res_max;
}

void NonlinearGS::enrich() {
  for (int i = 0; i < linear_ops.size(); ++i) {
    inner_products_x[i].push_back(inner_products_one);
    linear_ops[i]->enrich();
    linear_ops[i]->normalize();
    if (i > 0) {
      linear_ops[i]->get_inner_products(inner_products_x[i], 
                                        inner_products_b[i]);
    }
  }
}

void NonlinearGS::set_inner_products() {
  for (int i = 0; i < linear_ops.size(); ++i)
    linear_ops[i]->get_inner_products(inner_products_x[i], inner_products_b[i]);
}

void NonlinearGS::finalize() {
  const int num_modes = inner_products_x[0].size();
  std::cout << "num_modes " << num_modes << std::endl;
  for (int i = 0; i < linear_ops.size(); ++i) {
    inner_products_all_x[i].push_back(inner_products_x[i]);
    inner_products_all_b[i].push_back(inner_products_b[i]);
    AssertDimension(inner_products_all_x[i].size(), num_modes);
    AssertDimension(inner_products_all_b[i].size(), num_modes);
    for (int m_row = 0; m_row < num_modes - 1; ++m_row) {
      inner_products_all_x[i][m_row].push_back(inner_products_one);
      std::cout << inner_products_all_x[i][m_row].size() << std::endl;
      AssertDimension(inner_products_all_x[i][m_row].size(), num_modes);
      linear_ops[i]->get_inner_products(
          inner_products_all_x[i][m_row], inner_products_all_b[i][m_row],
          m_row, num_modes-1);
    }
  }
}

void NonlinearGS::update() {
  const int num_modes = inner_products_x[0].size();
  const int num_sources = inner_products_b[0].size();
  for (int i = 0; i < linear_ops.size(); ++i) {
    auto updatable = dynamic_cast<LinearUpdatableInterface*>(linear_ops[i]);
    if (updatable == NULL)
      continue;
    std::vector<std::vector<InnerProducts>> coefficients_x(
        num_modes, std::vector<InnerProducts>(num_modes, inner_products_one));
    std::vector<std::vector<double>> coefficients_b(
        num_modes, std::vector<double>(num_sources, 1.0));
    for (int j = 0; j < linear_ops.size(); ++j) {
      if (i == j)
        continue;
      for (int m_row = 0; m_row < num_modes; ++m_row) {
        for (int m_col = 0; m_col < num_modes; ++m_col)
          coefficients_x[m_row][m_col] *= inner_products_all_x[j][m_row][m_col];
        for (int s = 0; s < num_sources; ++s)
          coefficients_b[m_row][s] *= inner_products_all_b[j][m_row][s];
      }
    }
    updatable->update(coefficients_x, coefficients_b);
  }
}

void NonlinearGS::reweight() {
  
}

}  // namespace aether::pgd::sn