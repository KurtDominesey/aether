#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>

#include "mesh/mesh.h"
#include "gtest/gtest.h"

#include "svg_flags.h"

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
  std::vector<double> radii{0.4095, 0.4750, 0.54};
  std::vector<int> materials;
  for (int i = 0; i < radii.size() + 1; ++i)
    materials.push_back(i);
  mesh_symmetric_quarter_pincell(triangulation, radii, 0.63, materials);
  dealii::GridOut grid_out;
  grid_out.set_flags(svg_flags());
  for (int r = 0; r < 4; ++r) {
    std::string filename = "symmetric_quarter_pincell_r" 
                           + std::to_string(r) + ".svg";
    std::ofstream file(filename);
    grid_out.write_svg(triangulation, file);
    triangulation.refine_global(1);
  }
}

TEST(MeshTest, Refinement) {
  dealii::Triangulation<2> mesh;
  std::vector<double> radii{0.4095, 0.4180, 0.4750, 0.4850, 0.54};
  std::vector<int> materials(radii.size()+1);
  std::iota(materials.begin(), materials.end(), 0);
  mesh_symmetric_quarter_pincell(mesh, radii, 0.63, materials);
  double target = 7.5e-2;
  dealii::Vector<double> measures;
  do {
    measures.reinit(mesh.n_active_cells());
    int c = 0;
    for (auto cell = mesh.begin_active(); cell != mesh.end(); ++cell, ++c) {
      measures[c] = 
          std::max(cell->extent_in_direction(1), cell->extent_in_direction(0));
      if (measures[c] < target)
        continue;
      int i = cell->extent_in_direction(1) > cell->extent_in_direction(0);
      cell->set_refine_flag(dealii::RefinementCase<2>::cut_axis(i));
    }
    mesh.execute_coarsening_and_refinement();
  } while (*std::max_element(measures.begin(), measures.end()) > target);
  dealii::GridOut grid_out;
  grid_out.set_flags(svg_flags());
  std::string filename = "refined.svg";
  std::ofstream file(filename);
  grid_out.write_svg(mesh, file);
}

TEST(MeshTest, C5G7) {
  dealii::Triangulation<2> triangulation;
  mesh_symmetric_quarter_pincell(triangulation, {0.54}, 0.63, {1, 0});
  dealii::GridOut grid_out;
  grid_out.set_flags(svg_flags());
  for (int r = 0; r < 5; ++r) {
    std::string filename = "c5g7_r" + std::to_string(r) + ".svg";
    std::ofstream file(filename);
    grid_out.write_svg(triangulation, file);
    triangulation.refine_global(1);
  }
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
  triangulation.refine_global(2);
  dealii::GridOut grid_out;
  std::string filename = "pincell.svg";
  std::ofstream file(filename);
  grid_out.write_svg(triangulation, file);
}

TEST(MeshTest, SymmetricQuarterPincellRefined) {
  dealii::Triangulation<2> triangulation;
  std::vector<double> radii{0.4095, 0.4180, 0.4750, 0.4850, 0.5400};
  std::vector<int> materials;
  for (int i = 0; i < radii.size() + 1; ++i)
    materials.push_back(i);
  std::vector<int> max_levels{4, 2, 2, 2, 2, 4};
  mesh_symmetric_quarter_pincell(triangulation, radii, 0.63, materials);
  dealii::GridOut grid_out;
  grid_out.set_flags(svg_flags());
  refine_azimuthal(triangulation, 2);
  refine_radial(triangulation, 2, max_levels);
  std::string filename = "symmetric_quarter_pincell_refined.svg";
  std::ofstream file(filename);
  grid_out.write_svg(triangulation, file);
}

}  // namespace

}  // namespace aether