#include "base/block_block_vector.h"
#include "gtest/gtest.h"

namespace aether {

namespace {

template <typename Number>
class BlockBlockVectorTest : public ::testing::Test {};

using Numbers = ::testing::Types<double, float>;
TYPED_TEST_CASE(BlockBlockVectorTest, Numbers);

TYPED_TEST(BlockBlockVectorTest, Init) {
  using Number = TypeParam;
  int n_blocks = 5;
  int n_subblocks = 7;
  int block_size = 11;
  BlockBlockVector<Number> vector(n_blocks, n_subblocks, block_size);
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
    EXPECT_NE(dynamic_cast<dealii::BlockVector<Number>*>(&block), nullptr);
    for (int s = 0; s < n_subblocks; ++s) {
      auto& subblock = block.block(s);
      EXPECT_EQ(subblock.size(), block_size);
      EXPECT_NE(dynamic_cast<dealii::Vector<Number>*>(&subblock), nullptr);
    }
  }
}

TYPED_TEST(BlockBlockVectorTest, Assign) {
  using Number = TypeParam;
  BlockBlockVector<Number> vector(5, 7, 11);
  for (int i = 0; i < vector.size(); ++i)
    EXPECT_EQ(vector[i], 0);
  Number value = 3.14;
  vector = value;
  BlockBlockVector<Number> copied(vector);
  BlockBlockVector<Number> assigned;
  assigned = vector;
  for (int i = 0; i < vector.size(); ++i) {
    EXPECT_EQ(vector[i], value);
    EXPECT_EQ(copied[i], value);
    EXPECT_EQ(assigned[i], value);
  }
  EXPECT_EQ(copied, vector);
  EXPECT_EQ(assigned, vector);
}

}  // namespace

}  // namespace aether