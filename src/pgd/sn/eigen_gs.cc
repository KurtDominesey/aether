#include "pgd/sn/eigen_gs.h"

namespace aether::pgd::sn {

EigenGS::EigenGS(std::vector<LinearInterface*>& linear_ops,
                 int num_materials, int num_legendre)
    : NonlinearGS(linear_ops, num_materials, num_legendre, 0) {
  for (LinearInterface* op : linear_ops)
    eigen_ops.push_back(dynamic_cast<EigenInterface*>(op));
}

double EigenGS::initialize_iteratively(double tol) {
  std::cout << "initializing\n";
  double k0 = 0;
  double k1 = 1;
  this->enrich();
  std::cout << "enriched\n";
  AssertDimension(this->inner_products_x[0].size(), 1);
  std::vector<InnerProducts> coefficients(this->inner_products_x[0].size(),
                                          this->inner_products_x[0].back());
  std::vector<double> _;
  while (std::abs(k1 - k0) > tol) {
    k0 = k1;
    for (int i = 0; i < eigen_ops.size(); ++i) {
      this->set_coefficients(i, coefficients, _);
      k1 = eigen_ops[i]->step_eigenvalue(coefficients[0]);
      std::cout << k1 << std::endl;
      linear_ops[i]->get_inner_products(inner_products_x[i], _);
    }
  }
  this->finalize();
  return update();
}

double EigenGS::initialize_guess() {
  this->enrich();
  this->finalize();
  return update();
}

double EigenGS::update() {
  const int num_modes = inner_products_x[0].size();
  double k = 0;
  for (int i = 0; i < eigen_ops.size(); ++i) {
    auto updatable = dynamic_cast<EigenUpdatableInterface*>(eigen_ops[i]);
    if (updatable == NULL)
      continue;
    std::vector<std::vector<InnerProducts>> coefficients(
        num_modes, std::vector<InnerProducts>(num_modes, inner_products_one));
    for (int j = 0; j < eigen_ops.size(); ++j) {
      if (i == j)
        continue;
      for (int m_row = 0; m_row < num_modes; ++m_row) {
        for (int m_col = 0; m_col < num_modes; ++m_col) {
          coefficients[m_row][m_col] *= inner_products_all_x[j][m_row][m_col];
        }
      }
    }
    k = updatable->update(coefficients);
    for (int j = 0; j < eigen_ops.size(); ++j) {
      if (j != i) {
        eigen_ops[j]->eigenvalue = k;
      }
    }
  }
  Assert(k != 0, dealii::ExcMessage("Eigenvalue must be nonzero"));
  std::cout << "NEW EIGENVALUE: " << k << std::endl;
  return k;
}

}