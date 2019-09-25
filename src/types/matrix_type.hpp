#ifndef AETHER_TYPES_MATRIX_TYPE_H_
#define AETHER_TYPES_MATRIX_TYPE_H_

#include <deal.II/lac/vector.h>

class MatrixType {
 public:
  virtual void vmult(dealii::Vector<double> &src,
                     dealii::Vector<double> &dst) = 0;
  virtual void Tvmult(dealii::Vector<double> &src,
                      dealii::Vector<double> &dst) = 0;
};

#endif // AETHER_TYPES_MATRIX_TYPE_H_