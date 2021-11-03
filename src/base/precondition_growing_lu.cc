#include "base/precondition_growing_lu.h"

namespace aether {

template <typename number>
void PreconditionGrowingLU<number>::initialize(
    const dealii::LAPACKFullMatrix_<number> &a) {
  AssertDimension(a.m(), a.n());
  num_blocks = 1;
  block_size = a.m();
  matrix = a;
  matrix.compute_lu_factorization();
}

template <typename number>
void PreconditionGrowingLU<number>::grow(
    dealii::LAPACKFullMatrix_<number> &a12,
    dealii::LAPACKFullMatrix_<number> &a21,
    dealii::LAPACKFullMatrix_<number> &a22) {
  AssertDimension(a22.m(), block_size);
  AssertDimension(a22.n(), block_size);
  AssertDimension(a12.m(), block_size*num_blocks);
  AssertDimension(a12.n(), block_size);
  AssertDimension(a21.m(), block_size);
  AssertDimension(a21.n(), block_size*num_blocks);
  const dealii::types::blas_int bs = num_blocks * block_size;
  const dealii::types::blas_int s = block_size;
  const dealii::types::blas_int one = 1;
  const number neg1 = -1.;
  const number pos1 = 1.;
  // Pivot rows of A_12
  laswp(&s, a12.values.data(), &bs, &one, &bs, matrix.ipiv.data(), &one);
  // Find X_21 such that: X_21 U_11 = A_21, overwriting X_21 on A_21
  trsm("R", "U", "N", "N", &s, &bs, &pos1, matrix.values.data(), &bs, 
       a21.values.data(), &s);
  // Find X_12 such that: L_11 X_12 = A_12, overwriting X_12 on A_12
  trsm("L", "L", "N", "U", &bs, &s, &pos1, matrix.values.data(), &bs, 
       a12.values.data(), &bs);
  // Compute Schur complement, A_22 := A_22 - A_21 A_12
  dealii::gemm("N", "N", &s, &s, &bs, &neg1, a21.values.data(), &s, 
               a12.values.data(), &bs, &pos1, a22.values.data(), &s);
  // A_22 = L_22 U_22
  a22.compute_lu_factorization();
  // Pivot rows of A_21
  laswp(&bs, a21.values.data(), &s, &one, &s, a22.ipiv.data(), &one);
  // Copy A_12, A_21, and A_22 into A (= L U)
  matrix.grow_or_shrink(matrix.m()+block_size);
  int last = num_blocks * block_size;
  num_blocks += 1;
  for (int b = 0; b < num_blocks; ++b) {
    int bb = b * block_size;
    for (int i = 0; i < block_size; ++i) {
      for (int j = 0; j < block_size; ++j) {
        if (b < num_blocks - 1) {
          matrix(bb+i, last+j) = a12(bb+i, j);  // last block column
          matrix(last+i, bb+j) = a21(i, bb+j);  // last block row
        } else {
          matrix(last+i, last+j) = a22(i, j);  // last diagonal block
        }
      }
    }
  }
  // Add (and offset) pivot indices from A_22
  matrix.ipiv.reserve(matrix.ipiv.size()+a22.ipiv.size());
  for (int i = 0; i < a22.ipiv.size(); ++i)
    matrix.ipiv.push_back(last+a22.ipiv[i]);
}

template class PreconditionGrowingLU<double>;
// template class PreconditionGrowingLU<float>;

}  // namespace aether