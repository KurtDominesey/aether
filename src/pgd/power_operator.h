#ifndef AETHER_PGD_POWER_OPERATOR_H_
#define AETHER_PGD_POWER_OPERATOR_H_

#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/linear_operator.h>

#include "../types/matrix_type.h"

namespace pgd {

class PowerOperator : public MatrixType {
 template <typename T>
 using Separated = std::map<std::string, std::vector<T> >;

 public:
  PowerOperator (std::vector<MatrixType>& operators);
  void vmult(dealii::BlockVector<double>& dst, 
             dealii::BlockVector<double>& src);
  void Tvmult(dealii::BlockVector<double>& dst, 
              dealii::BlockVector<double>& src);
  unsigned int n_block_rows() const;
  unsigned int n_block_cols() const;
  std::vector < std::map<std::string, std::vector<double> > > coefficients_by_op;
  std::vector<double> eigenvalues;
  std::vector<dealii::LinearOperator<>>& operators;
};

void PowerOperator::vmult(dealii::BlockVector<double>& dst, 
                          dealii::BlockVector<double>& src) {
  std::map<std::string, std::vector<double>> coefficients;
  for (int i = 0; i < n_block_rows(); ++i) {
    dealii::SolverControl(100, 1e-6);
    operators[i].vmult(dst.block(i), src.block(i));
    coefficients.update(operators[i]);
  }
}

}

#endif  // AETHER_PGD_POWER_OPERATOR_H_
