#include "base/mgxs.h"
#include "gtest/gtest.h"

namespace aether {

namespace {

const std::string dir = "src/base/tests/";
const std::string file = "c5g7.h5";
const std::string temperature = "294K";
const std::vector<std::string> materials = {"water", "uo2"};

TEST(MgxsTest, Read) { 
  const Mgxs mgxs = read_mgxs(
      dir+file, temperature, materials);
}

TEST(MgxsTest, ReadWrite) {
  const std::string name = dir + "readwrite.h5";
  const Mgxs mgxs = read_mgxs(dir+file, temperature, materials);
  write_mgxs(mgxs, name, temperature, materials);
  const Mgxs mgxs_test = read_mgxs(name, temperature, materials);
  const int num_groups = mgxs.total.size();
  for (int j = 0; j < materials.size(); ++j) {
    for (int g = 0; g < num_groups; ++g) {
      EXPECT_DOUBLE_EQ(mgxs.total[g][j], mgxs_test.total[g][j]);
      for (int gp = 0; gp < num_groups; ++gp) {
        EXPECT_DOUBLE_EQ(mgxs.scatter[g][gp][j], mgxs_test.scatter[g][gp][j]);
      }
    }
  }
}

}  // namespace

}  // namespace aether