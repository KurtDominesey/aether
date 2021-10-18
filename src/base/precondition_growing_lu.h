#ifndef AETHER_BASE_PRECONDITION_GROWING_LU_H_
#define AETHER_BASE_PRECONDITION_GROWING_LU_H_

#include <deal.II/lac/lapack_full_matrix.h>

namespace aether {

/**
 * Preconditioner based on the direct, blockwise inversion of a growing matrix.
 * 
 * Given a 2x2 block matrix [[A, B], [C, D]], the inverse can be directly
 * computed from the inverses of A and the Schur complement of A. This class
 * initially inverts a matrix A and recomputes the inverse as the matrix is
 * incrementally grown by appending blocks B, C, and D in calls to grow.
 */
template <typename number>
class PreconditionGrowingLU : public dealii::Subscriptor {
 public:
  void initialize(const dealii::LAPACKFullMatrix<number> &a);
  void grow(const dealii::LAPACKFullMatrix<number> &b, 
            const dealii::LAPACKFullMatrix<number> &c, 
            const dealii::LAPACKFullMatrix<number> &d);
  dealii::LAPACKFullMatrix<number> inverse;

 protected:
  int num_blocks, block_size;
  dealii::LAPACKFullMatrix<number> a_inv, schur, aux0, aux1;
};

}  // namespace aether

#endif  // AETHER_BASE_PRECONDITION_GROWING_LU_H_