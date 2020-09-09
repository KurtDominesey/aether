#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_out.h>

#include "mesh/mesh.h"
#include "gtest/gtest.h"

#include "svg_flags.h"

namespace aether {

namespace {

TEST(MeshTest, MoxAssembly) {
  dealii::Triangulation<2> mesh;
  mesh_mox_assembly(mesh);
  mesh.refine_global(0);
  dealii::GridOut grid_out;
  grid_out.set_flags(svg_flags());
  std::string filename = "mox_assembly.svg";
  std::ofstream file(filename);
  grid_out.write_svg(mesh, file);
}

}  // namespace

}  // namespace aether