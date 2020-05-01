// #include "../mgxs.h"
#include "base/mgxs.cc"
#include "gtest/gtest.h"

namespace aether {

namespace {

TEST(MgxsTest, Read) { 
  const Mgxs mgxs = read_mgxs(
      "src/base/tests/c5g7.h5", "294K", {"water", "uo2"});
}

TEST(MgxsTest, ReadWrite) {
  const std::string dir = "src/base/tests/";
  const std::string name = dir+"MGXS_TEST.h5";
  const std::string temperature = "294K";
  const std::vector<std::string> materials = {"water", "mox43"};
  const Mgxs mgxs = read_mgxs(dir+"gold.h5", temperature, materials);
  write_mgxs(mgxs, name, temperature, materials);
  // Mgxs mgxs_test(mgxs.total.size(), materials.size(), 1);
  // read_mgxs(mgxs_test, name, temperature, materials);
  const Mgxs mgxs_test = read_mgxs(name, temperature, materials);
  const int num_groups = mgxs.total.size();
  const int num_materials = materials.size();
  for (const Mgxs &mgxs_p : {mgxs, mgxs_test}) {
    std::cout << "\n";
    for (int g = 0; g < num_groups; ++g) {
      // std::cout << mgxs_p.total[g][0];
      for (int gp = 0; gp < num_groups; ++gp) {
        std::cout << mgxs_p.scatter[g][gp][1] << " ";
      }
      std::cout << "\n";
    }
  }
  for (int j = 0; j < num_materials; ++j) {
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