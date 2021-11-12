#ifndef AETHER_BASE_PRECONDITION_BLOCK_GROWING_H_
#define AETHER_BASE_PRECONDITION_BLOCK_GROWING_H_

#include "base/lapack_full_matrix.h"

namespace aether {

/**
 * Block preconditioner where the diagonal blocks grow incrementally.
 * 
 * Analogous to deal.II's PreconditionBlock, but where the diagonal blocks are
 * represented as growing LU decompositions (that is, instances of 
 * PreconditionGrowingLU).
 */
template <typename MatrixType, typename number>
class PreconditionBlockGrowingLU : public dealii::Subscriptor {
 public:
  void initialize(const dealii::LAPACKFullMatrix_<number> &a);
  void grow(dealii::LAPACKFullMatrix_<number> &a12,
            dealii::LAPACKFullMatrix_<number> &a21,
            dealii::LAPACKFullMatrix_<number> &a22);
  void vmult(dealii::Vector<number> &dst, const dealii::Vector<number> &src) 
    const;
  MatrixType *matrix;
 protected:
  std::vector<PreconditionGrowingLU<number>> blocks;
  int num_subblocks;
};

template <typename MatrixType, typename number>
void PreconditionBlockGrowingLU<MatrixType, number>::initialize(
    const dealii::LAPACKFullMatrix_<number> &a) {
  blocks.emplace_back();
  blocks.back().initialize(a);
}

template <typename MatrixType, typename number>
void PreconditionBlockGrowingLU<MatrixType, number>::grow(
    dealii::LAPACKFullMatrix_<number> &a12,
    dealii::LAPACKFullMatrix_<number> &a21,
    dealii::LAPACKFullMatrix_<number> &a22) {
  if (true /*blocks.back().num_blocks < num_subblocks*/) {
    blocks.back().grow(a12, a21, a22);
  }
}

template <typename MatrixType, typename number>
void PreconditionBlockGrowingLU<MatrixType, number>::vmult(
    dealii::Vector<number> &dst, const dealii::Vector<number> &src) const {
  Assert(matrix != nullptr, dealii::ExcInvalidState());
  dealii::Vector<number> dst_b;
  dealii::Vector<number> src_b;
  int offset = 0;
  for (int b = 0; b < blocks.size(); ++b) {
    int size = blocks[b].matrix.m();
    dst_b.grow_or_shrink(size);
    src_b.grow_or_shrink(size);
    for (int i = 0; i < size; ++i) {
      const typename MatrixType::const_iterator row_end = matrix->end(i);
      typename MatrixType::const_iterator entry = matrix->begin(i);
      src_b[i] = src[offset+i];
      for (; entry != row_end; ++entry) {
        auto j = entry->column();
        if (j >= offset)  // not in lower triangle
          break;
        AssertThrow(false, dealii::ExcNotImplemented());
        src_b[i] -= entry->value() * dst(j);
      }
    }
    blocks[b].vmult(dst_b, src_b);
    for (int i = 0; i < size; ++i) {
      dst[offset+i] = dst_b[i];
    }
    offset += size;
  }
}

}

#endif  // AETHER_BASE_PRECONDITION_BLOCK_GROWING_H_