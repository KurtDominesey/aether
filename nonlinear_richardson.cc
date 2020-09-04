#include "pgd/sn/nonlinear_richardson.h"

namespace aether::pgd::sn {

NonlinearRichardson::NonlinearRichardson(
    std::vector<LinearInterface*> &linear_ops, 
    int num_materials, int num_legendre, int num_sources)
    : linear_ops(linear_ops),
      inner_products_x(linear_ops.size()),
      inner_products_b(linear_ops.size(), std::vector<double>(num_sources)),
      inner_products_one(num_materials, num_legendre) {
  inner_products_one = 1;
}

void NonlinearRichardson::vmult(dealii::BlockVector<double> dst,
                                const dealii::BlockVector<double> src) {
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
    linear_ops[i]->get_residuals(res, res_sq);
    auto& A = linear_ops[i]->get_operator();
    auto& b = linear_ops[i]->get_source();
    linear_ops[i]->vmult(dst, src, coefficients_x, coefficients_b);
  }
}

}  // namespace aether::pgd::sn