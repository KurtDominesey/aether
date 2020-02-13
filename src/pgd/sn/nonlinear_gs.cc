#include "pgd/sn/nonlinear_gs.h"

namespace aether::pgd::sn {

NonlinearGS::NonlinearGS(std::vector<LinearInterface*> &linear_ops,
                         int num_materials, int num_legendre, int num_sources)
    : linear_ops(linear_ops),
      inner_products_x(linear_ops.size()),
      inner_products_b(linear_ops.size(), std::vector<double>(num_sources)),
      inner_products_one(num_materials, num_legendre) {
  inner_products_one = 1;
}

void NonlinearGS::step(dealii::BlockVector<double> x, 
                       const dealii::BlockVector<double> b) {
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
    linear_ops[i]->step(x, b, coefficients_x, coefficients_b);
    if (i > 0)
      linear_ops[i]->normalize();
    linear_ops[i]->get_inner_products(inner_products_x[i], inner_products_b[i]);
  }
}

void NonlinearGS::enrich() {
  for (int i = 0; i < linear_ops.size(); ++i) {
    inner_products_x[i].push_back(inner_products_one);
    linear_ops[i]->enrich();
    if (i > 0) {
      linear_ops[i]->normalize();
      linear_ops[i]->get_inner_products(inner_products_x[i], 
                                        inner_products_b[i]);
    }
  }
}

}  // namespace aether::pgd::sn