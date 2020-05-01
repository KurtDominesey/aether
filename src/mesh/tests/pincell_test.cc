#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_out.h>

#include "mesh/mesh.cc"
#include "gtest/gtest.h"

namespace aether {

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

TEST(MeshTest, SymmetricQuarterPincell) {
  dealii::Triangulation<2> triangulation;
  mesh_symmetric_quarter_pincell(triangulation, {0.44, 0.54}, 0.63, {1, 1, 0});
  triangulation.refine_global(2);
  dealii::GridOut grid_out;
  std::string filename = "symmetric_quarter_pincell.svg";
  std::ofstream file(filename);
  grid_out.write_svg(triangulation, file);
}

TEST(MeshTest, EighthPincell) {
  dealii::Triangulation<2> triangulation;
  mesh_eighth_pincell_ul(triangulation, {0.44, 0.54}, 0.63, {1, 1, 0});
  triangulation.refine_global(3);
  dealii::GridOut grid_out;
  std::string filename = "eighth_pincell_ul.svg";
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

TEST(MeshTest, MoxAssembly) {
  dealii::Triangulation<2> mesh;
  mesh_mox_assembly(mesh);
  mesh.refine_global(0);
  dealii::GridOut grid_out;
  std::string filename = "mox_assembly.svg";
  std::ofstream file(filename);
  grid_out.write_svg(mesh, file);
}

}  // namespace

}  // namespace aether