#include "base/precondition_growing_lu.h"
#include "base/precondition_block_growing_lu.h"
#include "gtest/gtest.h"

namespace aether {

namespace {

class PreconditionGrowingLUTest : public ::testing::Test {
 protected:
  dealii::LAPACKFullMatrix_<double> a, b, c, d, eye;
  PreconditionGrowingLU<double> a_lu;
};

TEST_F(PreconditionGrowingLUTest, GrowThrice) {
  int block_size = 5;
  a.reinit(block_size);
  eye.reinit(block_size);
  for (int i = 0; i < block_size; ++i)
    a(i, i) = i + 1;
  a_lu.initialize(a);
  eye = a;
  a_lu.matrix.solve(eye);
  for (int i = 0; i < eye.m(); ++i)
    for (int j = 0; j < eye.n(); ++j)
      EXPECT_NEAR(eye(i, j), i == j, 1e-14);
  for (int num_blocks = 1; num_blocks <= 3; ++num_blocks) {
    int size = num_blocks * block_size;
    b.reinit(size, block_size);
    c.reinit(block_size, size);
    d.reinit(block_size);
    a.grow_or_shrink(a.m()+block_size);
    eye.reinit(eye.m()+block_size);
    int last = num_blocks * block_size;
    for (int bl = 0; bl <= num_blocks; ++bl) {
      int off = bl * block_size;
      for (int i = 0; i < block_size; ++i) {
        for (int j = 0; j < block_size; ++j) {
          if (bl < num_blocks) {
            b(off+i, j) = (bl+1) * (i+j+1);
            c(i, off+j) = -(bl+1) * (i+j+1);
            a(off+i, last+j) = b(off+i, j);
            a(last+i, off+j) = c(i, off+j);
          } else {
            d(i, j) = (i == j) * std::pow(i+2, 2);
            a(last+i, last+j) = d(i, j);
          }
        }
      }
    }
    a_lu.grow(b, c, d);
    eye = a;
    a_lu.matrix.solve(eye);
    for (int i = 0; i < eye.m(); ++i)
      for (int j = 0; j < eye.n(); ++j)
        EXPECT_NEAR(eye(i, j), i == j, 1e-14);
  }
}

class PreconditionBlockGrowingLUTest : public ::testing::Test {
 protected:
  dealii::Vector<double> v, av, iv;
  dealii::LAPACKFullMatrix_<double> a, d;
  PreconditionBlockGrowingLU<dealii::LAPACKFullMatrix_<double>, double> a_jac;
};

TEST_F(PreconditionBlockGrowingLUTest, Jacobi) {
  int block_size = 5;
  v.reinit(block_size);
  av.reinit(block_size);
  iv.reinit(block_size);
  a.reinit(block_size);
  for (int i = 0; i < block_size; ++i) {
    a(i, i) = i + 1;
    v = std::pow(i, 2);
  }
  a.vmult(av, v);
  a_jac.initialize(a);
  a_jac.matrix = &a;
  a_jac.vmult(iv, av);
  for (int i = 0; i < v.size(); ++i)
    EXPECT_NEAR(iv[i], v[i], 1e-14);
  for (int num_blocks = 1; num_blocks <= 3; ++num_blocks) {
    int size = num_blocks * block_size;
    d.reinit(block_size);
    a.grow_or_shrink(a.m()+block_size);
    v.grow_or_shrink(v.size()+block_size);
    av.reinit(v.size());
    iv.reinit(v.size());
    int last = num_blocks * block_size;
    for (int i = 0; i < block_size; ++i) {
      for (int j = 0; j < block_size; ++j) {
        d(i, j) = (i == j) * std::pow(i+2, 2);
        a(last+i, last+j) = d(i, j);
      }
    }
    a.vmult(av, v);
    a_jac.initialize(d);
    a_jac.vmult(iv, av);
    for (int i = 0; i < v.size(); ++i)
      EXPECT_NEAR(iv[i], v[i], 1e-14);
  }
}

TEST_F(PreconditionBlockGrowingLUTest, GaussSeidel) {
  dealii::LAPACKFullMatrix_<double> c;
  int block_size = 5;
  v.reinit(block_size);
  av.reinit(block_size);
  iv.reinit(block_size);
  a.reinit(block_size);
  for (int i = 0; i < block_size; ++i) {
    a(i, i) = i + 1;
    v = std::pow(i, 2);
  }
  a.vmult(av, v);
  a_jac.initialize(a);
  a_jac.matrix = &a;
  a_jac.vmult(iv, av);
  for (int i = 0; i < v.size(); ++i)
    EXPECT_NEAR(iv[i], v[i], 1e-14);
  for (int num_blocks = 1; num_blocks <= 3; ++num_blocks) {
    int size = num_blocks * block_size;
    d.reinit(block_size);
    c.reinit(block_size, size);
    a.grow_or_shrink(a.m()+block_size);
    v.grow_or_shrink(v.size()+block_size);
    av.reinit(v.size());
    iv.reinit(v.size());
    int last = num_blocks * block_size;
    for (int bl = 0; bl <= num_blocks; ++bl) {
      int off = bl * block_size;
      for (int i = 0; i < block_size; ++i) {
        for (int j = 0; j < block_size; ++j) {
          if (bl < num_blocks) {
            c(i, off+j) = -(bl+1) * (i+j+1);
            a(last+i, off+j) = c(i, off+j);
          } else {
            d(i, j) = std::pow(i+j+2, 2);
            a(last+i, last+j) = d(i, j);
          }
        }
      }
    }
    a.print_formatted(std::cout);
    a.vmult(av, v);
    a_jac.initialize(d);
    a_jac.vmult(iv, av);
    for (int i = 0; i < v.size(); ++i)
      EXPECT_NEAR(iv[i], v[i], 1e-14);
  }
}

}  // namespace

}  // namespace aether