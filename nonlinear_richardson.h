#ifndef AETHER_PGD_NONLINEAR_RICHARDSON_H_
#define AETHER_PGD_NONLINEAR_RICHARDSON_H_

#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_vector.h>

#include "pgd/sn/linear_interface.h"
#include "pgd/sn/inner_products.h"

namespace aether::pgd::sn {

class NonlinearRichardson {
 public:
  NonlinearRichardson(std::vector<LinearInterface*> &linear_ops,
                      int num_materials, int num_legendre, int num_sources);
  void vmult(dealii::BlockVector<double> dst,
             const dealii::BlockVector<double> src);
  void enrich();
  std::vector<std::vector<InnerProducts>> inner_products_x;
  std::vector<std::vector<double>> inner_products_b;
 protected:
  std::vector<LinearInterface*> &linear_ops;
  InnerProducts inner_products_one;
};

}  // namespace aether::pgd::sn

#endif  // AETHER_PGD_NONLINEAR_RICHARDSON_H_