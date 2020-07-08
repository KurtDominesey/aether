#ifndef AETHER_PGD_SN_LINEAR_INTERFACE_H_
#define AETHER_PGD_SN_LINEAR_INTERFACE_H_

#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_vector.h>

#include "pgd/sn/inner_products.h"

namespace aether::pgd::sn {

class LinearInterface {
 public:
  void virtual vmult(dealii::BlockVector<double> &dst, 
                     const dealii::BlockVector<double> &src,
                     std::vector<InnerProducts> coefficients_x,
                     std::vector<double> coefficients_b) = 0;
  double virtual step(dealii::Vector<double> &delta,
                    const dealii::BlockVector<double> &b,
                    std::vector<InnerProducts> coefficients_x,
                    std::vector<double> coefficients_b,
                    double omega = 1.) = 0;
  void virtual take_step(const double factor, 
                         const dealii::Vector<double> &delta) = 0;
  void virtual get_inner_products(std::vector<InnerProducts> &inner_products_x,
                                  std::vector<double> &inner_products_b) = 0;
  void virtual get_inner_products(std::vector<InnerProducts> &inner_products_x,
                                  std::vector<double> &inner_products_b,
                                  const int m_row, const int m_col_start) = 0;
  void virtual get_inner_products_x(std::vector<InnerProducts> &inner_products_x,
                                    const dealii::Vector<double> &left) = 0;
  void virtual get_inner_products_x(InnerProducts &inner_products_x,
                                    const dealii::Vector<double> &left,
                                    const dealii::Vector<double> &right) = 0;
  void virtual get_inner_products_x(InnerProducts &inner_products_x,
                                    const dealii::Vector<double> &right) = 0;
  void virtual get_inner_products_b(std::vector<double> &inner_products_b,
                                    const dealii::Vector<double> &left) = 0;
  double virtual line_search(
      std::vector<double> &c,
      const dealii::Vector<double> &step,
      const InnerProducts &coefficients_x_mode_step,
      const InnerProducts &coefficients_x_step_step,
      std::vector<InnerProducts> coefficients_x_mode_mode,
      std::vector<InnerProducts> coefficients_x_step_mode,
      const std::vector<double> &coefficients_b_mode,
      const std::vector<double> &coefficients_b_step) = 0;
  double virtual get_residual(std::vector<InnerProducts> coefficients_x,
                              std::vector<double> coefficients_b) = 0;
  double virtual enrich(const double factor) = 0;
  double virtual normalize() = 0;
  void virtual scale(double factor) = 0;
};

}  // namespace aether::pgd::sn

#endif  // AETHER_PGD_SN_LINEAR_OP_H_