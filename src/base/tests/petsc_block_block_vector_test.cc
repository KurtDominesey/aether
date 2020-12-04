#include "base/petsc_block_block_vector.h"
#include "gtest/gtest.h"

namespace aether {

namespace {

TEST(PETScBlockBlockVectorTest, Init) {
  int n_blocks = 5;
  int n_subblocks = 7;
  int block_size = 11;
  PETScWrappers::MPI::BlockBlockVector vector(
    n_blocks, n_subblocks, MPI_COMM_WORLD, block_size, block_size);
  EXPECT_EQ(vector.n_blocks(), n_blocks);
  EXPECT_EQ(vector.size(), n_blocks*n_subblocks*block_size);
  EXPECT_EQ(vector.get_block_indices().size(), n_blocks);
  EXPECT_EQ(vector.get_block_indices().total_size(), vector.size());
  for (int b = 0; b < n_blocks; ++b) {
    auto& block = vector.block(b);
    EXPECT_EQ(block.n_blocks(), n_subblocks);
    EXPECT_EQ(block.size(), n_subblocks*block_size);
    EXPECT_EQ(block.get_block_indices().size(), n_subblocks);
    EXPECT_EQ(block.get_block_indices().total_size(), block.size());
    EXPECT_NE(dynamic_cast<dealii::PETScWrappers::MPI::BlockVector*>(&block), 
              nullptr);
    for (int s = 0; s < n_subblocks; ++s) {
      auto& subblock = block.block(s);
      EXPECT_EQ(subblock.size(), block_size);
      EXPECT_NE(dynamic_cast<dealii::PETScWrappers::MPI::Vector*>(&subblock), 
                nullptr);
    }
  }
}

TEST(PETScBlockBlockVectorTest, Assign) {
  PETScWrappers::MPI::BlockBlockVector vector(5, 7, MPI_COMM_WORLD, 11, 11);
  for (int i = 0; i < vector.size(); ++i)
    EXPECT_EQ(vector[i], 0);
  double value = 3.14;
  vector = value;
  for (int i = 0; i < vector.size(); ++i) {
    EXPECT_EQ(vector[i], value);
    vector[i] = i;
  }
  vector.compress(dealii::VectorOperation::insert);
  PETScWrappers::MPI::BlockBlockVector copied(vector);
  PETScWrappers::MPI::BlockBlockVector assigned;
  vector.compress(dealii::VectorOperation::insert);
  assigned = vector;
  for (int i = 0; i < vector.size(); ++i) {
    EXPECT_EQ(copied[i], i);
    EXPECT_EQ(assigned[i], i);
  }
  EXPECT_EQ(copied, vector);
  EXPECT_EQ(assigned, vector);
}

}  // namespace

}  // namespace aether