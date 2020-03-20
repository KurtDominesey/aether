#ifndef AETHER_PGD_SN_LINEAR_UPDATABLE_INTERFACE_H_
#define AETHER_PGD_SN_LINEAR_UPDATABLE_INTERFACE_H_

#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_vector.h>

#include "pgd/sn/inner_products.h"
#include "pgd/sn/linear_interface.h"

namespace aether::pgd::sn {

class LinearUpdatableInterface : public LinearInterface {
 public:
  virtual void update(std::vector<std::vector<InnerProducts>> coefficients_x,
                      std::vector<std::vector<double>> coefficients_b) = 0;
};

}  // namespace aether::pgd::sn

#endif  // AETHER_PGD_SN_LINEAR_UPDATABLE_INTERFACE_H_