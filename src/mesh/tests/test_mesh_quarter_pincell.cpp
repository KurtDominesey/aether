#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_out.h>

#include "mesh/mesh.hpp"
#include "gtest/gtest.h"

namespace {

TEST(MeshTest, QuarterPincell) {
  dealii::Triangulation<2> triangulation;
  mesh_quarter_pincell(triangulation, {0.54, 0.6}, 0.63, {1, 1, 0});
  triangulation.refine_global(2);
  dealii::GridOut grid_out;
  std::string filename = "quarter_pincell.svg";
  std::ofstream file(filename);
  grid_out.write_svg(triangulation, file);
}

}