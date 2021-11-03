#ifndef AETHER_BASE_PRECONDITION_GROWING_LU_H_
#define AETHER_BASE_PRECONDITION_GROWING_LU_H_

#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/lapack_templates.h>

#include "base/lapack_full_matrix.h"
#include "base/lapack_templates.h"

namespace aether {

/**
 * Preconditioner based on the LU decomposition of a growing matrix.
 * 
 * Given a 2x2 block matrix [[A_11, A_12], [A_21, A_22]], the LU factorization
 * can be computed from that of A_11 and the Schur complement of A_11. This
 * class initially factorizes a matrix A and recomputes the factorization as the
 * matrix is incrementally grown by appending blocks A_12, A_21, and A_22 in
 * calls to grow.
 */
template <typename number>
class PreconditionGrowingLU : public dealii::Subscriptor {
 public:
  void initialize(const dealii::LAPACKFullMatrix_<number> &a);
  void grow(dealii::LAPACKFullMatrix_<number> &a12, 
            dealii::LAPACKFullMatrix_<number> &a21, 
            dealii::LAPACKFullMatrix_<number> &a22);
  dealii::LAPACKFullMatrix_<number> matrix;

 protected:
  int num_blocks, block_size;
};

}  // namespace aether

#endif  // AETHER_BASE_PRECONDITION_GROWING_LU_H_