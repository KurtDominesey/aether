#ifndef AETHER_PGD_FIXED_OPERATOR_H_
#define AETHER_PGD_FIXED_OPERATOR_H_

#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/solver_control.h>

#include "../types/matrix_type.hpp"
#include "cache.hpp"

namespace pgd {

class FixedOperator : public MatrixType {
  template <typename T>
  using Separated = std::map<char, std::vector<T> >;

 public:
  virtual dealii::LinearOperator<> inverse(std::vector<double> eigenvalues, 
                                           Separated<double> coefficients,
                                           Cache cache,
                                           dealii::SolverControl solver_control) 
                                           = 0;
  virtual double compute_inner_product(int m_trial, int m_test,
                                       std::string term, int summand) = 0;
  virtual void compute_last_inner_products();
  virtual void compute_all_inner_products();
  // Separated<std::vector<double> > coefficients;
  Cache cache;
};

}

#endif // AETHER_PGD_FIXED_OPERATOR_H_