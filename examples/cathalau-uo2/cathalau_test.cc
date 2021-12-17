#ifndef AETHER_EXAMPLES_CATHALAU_CATHALAU_TEST_H_
#define AETHER_EXAMPLES_CATHALAU_CATHALAU_TEST_H_

#include <deal.II/grid/grid_out.h>

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
  std::vector<double> radii = {0.4095, 0.4180, 0.4750, 0.4850, 0.5400};
  std::vector<int> regions = {2, 0, 3, 0, 4, 1};
  std::vector<int> max_levels = {2, 4, 4, 2, 2};

  void SetUp() override {
    this->testname = this->GetTestName();
    SetUp(false);
  }

  void SetUp(bool smear_void_gaps) {
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
        + materials[2] + "-v2/" + "mgxs-" + group_structure + ".h5";
    mgxs = std::make_unique<Mgxs>(num_groups, materials.size(), 1);
    read_mgxs(*mgxs, filename, "294K", materials, true);
    // smear void gaps
    if (smear_void_gaps) {
      for (int i = 1, d = 0; i < regions.size() - 1; i += 2, ++d) {
        int ii = i - d;
        double a0 = std::pow(radii[ii+1], 2) - std::pow(radii[ii], 2);
        double a1 = std::pow(radii[ii+1], 2) - std::pow(radii[ii-1], 2);
        double ratio = a0 / a1;
        int material = regions[ii];
        AssertThrow(material == 0, dealii::ExcInvalidState());
        for (int g = 0; g < mgxs->total.size(); ++g) {
          mgxs->total[g][material] *= ratio;
          for (int gp = 0; gp < mgxs->total.size(); ++gp) {
            mgxs->scatter[g][gp][material] *= ratio;
          }
        }
        radii.erase(radii.begin()+ii);
        regions.erase(regions.begin()+ii);
        std::cout << ratio << std::endl;
      }
    }
    quadrature = QPglc<dim_, qdim_>(4, 8);
    AssertDimension(regions.size(), radii.size()+1);
    mesh_symmetric_quarter_pincell(mesh, radii, pitch, regions);
    set_all_boundaries_reflecting(mesh);
  }

  void SetVolumes(std::vector<double> &volumes) {
    volumes.resize(materials.size());
    std::vector<double> areas(radii.size());  // ring areas
    for (int r = 0; r < radii.size(); ++r)
      areas[r]= dealii::numbers::PI * std::pow(radii[r], 2);
    volumes[regions.back()] = std::pow(2*pitch, 2) - areas.back();
    for (int r = 0; r < radii.size(); ++r)
      volumes[regions[r]] += areas[r] - (r == 0 ? 0 : areas[r-1]);
  }

  void PrintMesh() {
    dealii::GridOutFlags::Svg svg;
    svg.coloring = dealii::GridOutFlags::Svg::Coloring::material_id;
    svg.margin = false;
    svg.label_cell_index = false;
    svg.label_level_number = false;
    svg.label_level_subdomain_id = false;
    svg.label_material_id = false;
    svg.label_subdomain_id = false;
    svg.draw_colorbar = false;
    svg.draw_legend = false;
    dealii::GridOut grid_out;
    grid_out.set_flags(svg);
    std::string filename = "mesh_"+this->GetTestName()+".svg";
    std::ofstream file(filename);
    grid_out.write_svg(mesh, file);
  }
};

}  // namespace cathalau

#endif  // AETHER_EXAMPLES_CATHALAU_CATHALAU_TEST_H_