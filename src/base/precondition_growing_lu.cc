#include "base/precondition_growing_lu.h"

namespace aether {

template <typename number>
void PreconditionGrowingLU<number>::initialize(
    const dealii::LAPACKFullMatrix<number> &a) {
  AssertDimension(a.m(), a.n());
  num_blocks = 1;
  block_size = a.m();
  a_inv = a;
  a_inv.invert();
  inverse = a_inv;
}

template <typename number>
void PreconditionGrowingLU<number>::grow(
    const dealii::LAPACKFullMatrix<number> &b,
    const dealii::LAPACKFullMatrix<number> &c,
    const dealii::LAPACKFullMatrix<number> &d) {
  AssertDimension(d.m(), block_size);
  AssertDimension(d.n(), block_size);
  AssertDimension(b.m(), block_size*num_blocks);
  AssertDimension(b.n(), block_size);
  AssertDimension(c.m(), block_size);
  AssertDimension(c.n(), block_size*num_blocks);
  a_inv = inverse;
  // schur complement: S = D-CA^{-1}B
  aux0.reinit(a_inv.m(), b.n());
  schur = d;
  a_inv.mmult(aux0, b);
  aux0 *= -1;
  c.mmult(schur, aux0, true);
  schur.invert();
  // block A (square): A^{-1}+A^{-1}BS^{-1}CA^{-1}
  aux0.reinit(c.m(), a_inv.n());
  aux1.reinit(schur.m(), aux0.n());
  c.mmult(aux0, a_inv);
  schur.mmult(aux1, aux0);
  aux0.reinit(b.m(), aux1.n());
  b.mmult(aux0, aux1);
  a_inv.mmult(inverse, aux0, true);
  // grow inverse by one block
  inverse.grow_or_shrink(inverse.m()+block_size);
  int size = num_blocks * block_size;
  // block B (tall and skinny): -A^{-1}BS^{-1}
  aux0.reinit(b.m(), schur.n());
  aux1.reinit(a_inv.m(), aux0.n());
  b.mmult(aux0, schur);
  a_inv.mmult(aux1, aux0);
  for (int ni = 0; ni < size; ++ni) {
    for (int j = 0; j < block_size; ++j) {
      inverse(ni, size+j) = -aux1(ni, j);
    }
  }
  // block C (short and wide): -S^{-1}CA^{-1}
  aux0.reinit(c.m(), a_inv.n());
  aux1.reinit(schur.m(), aux0.n());
  c.mmult(aux0, a_inv);
  schur.mmult(aux1, aux0);
  for (int nj = 0; nj < size; ++nj) {
    for (int i = 0; i < block_size; ++i) {
      inverse(size+i, nj) = -aux1(i, nj);
    }
  }
  // block D (square): S^{-1}
  for (int i = 0; i < block_size; ++i) {
    for (int j = 0; j < block_size; ++j) {
      inverse(size+i, size+j) = schur(i, j);
    }
  }
  num_blocks += 1;
}

template class PreconditionGrowingLU<double>;
template class PreconditionGrowingLU<float>;

}  // namespace aether