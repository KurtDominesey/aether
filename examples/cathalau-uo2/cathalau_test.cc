#ifndef AETHER_EXAMPLES_CATHALAU_CATHALAU_TEST_H_
#define AETHER_EXAMPLES_CATHALAU_CATHALAU_TEST_H_

#include "../example_test.h"

namespace cathalau {

using namespace aether;
using namespace aether::sn;

static const int dim_ = 2;
static const int qdim_ = 2;

class CathalauTest : virtual public ExampleTest<dim_, qdim_> {
 protected:
  const double pitch = 0.63;
  std::string group_structure = "CASMO-70";
  std::vector<std::string> materials = {"void", "water", "uo2", "zr", "al"};
  const std::vector<double> radii = {0.4095, 0.4180, 0.4750, 0.4850, 0.5400};
  const std::vector<int> regions = {2, 0, 3, 0, 4, 1};

  void SetUp() override {
    // std::string group_structure;
    // auto this_with_param = 
    //     dynamic_cast<::testing::WithParamInterface<std::string>*>(this);
    // if (this_with_param != nullptr)
    //   group_structure = this_with_param->GetParam();
    // else
    //   group_structure = "CASMO-70";
    const int hyphen = group_structure.find("-");
    const std::string num_groups_str = group_structure.substr(hyphen+1);
    const int num_groups = std::stoi(num_groups_str);
    const std::string filename = 
        "/mnt/c/Users/kurt/Documents/projects/openmc-c5g7/" 
        + materials[2] + "/" + "mgxs-" + group_structure + ".h5";
    mgxs = std::make_unique<Mgxs>(num_groups, materials.size(), 1);
    read_mgxs(*mgxs, filename, "294K", materials, true);
    quadrature = QPglc<qdim_>(4, 4);
    AssertDimension(regions.size(), radii.size()+1);
    mesh_symmetric_quarter_pincell(mesh, radii, pitch, regions);
    set_all_boundaries_reflecting(mesh);
  }

  void SetVolumes() {
    std::vector<double> volumes(materials.size());
    std::vector<double> areas(radii.size());  // ring areas
    for (int r = 0; r < radii.size(); ++r)
      areas[r]= dealii::numbers::PI * std::pow(radii[r], 2);
    volumes[regions.back()] = std::pow(2*pitch, 2) - areas.back();
    for (int r = 0; r < radii.size(); ++r)
      volumes[regions[r]] += areas[r] - (r == 0 ? 0 : areas[r-1]);
  }
};

}  // namespace cathalau

#endif  // AETHER_EXAMPLES_CATHALAU_CATHALAU_TEST_H_