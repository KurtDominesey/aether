#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_out.h>

#include "mesh/mesh.h"
#include "gtest/gtest.h"

namespace {

TEST(MeshTest, QuarterPincell) {
  dealii::Triangulation<2> triangulation;
  mesh_quarter_pincell(triangulation, {0.44, 0.54}, 0.63, {1, 1, 0});
  triangulation.refine_global(3);
  dealii::GridOut grid_out;
  std::string filename = "quarter_pincell.svg";
  std::ofstream file(filename);
  grid_out.write_svg(triangulation, file);
}

TEST(MeshTest, Pincell) {
  dealii::Triangulation<2> triangulation;
  mesh_pincell(triangulation, {0.44, 0.54}, 0.63, {1, 1, 0});
  triangulation.refine_global(3);
  dealii::GridOut grid_out;
  std::string filename = "pincell.svg";
  std::ofstream file(filename);
  grid_out.write_svg(triangulation, file);
}

}