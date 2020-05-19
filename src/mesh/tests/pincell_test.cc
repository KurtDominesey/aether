#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>

#include "mesh/mesh.cc"
#include "gtest/gtest.h"

#include "base/tests/mgxs_test.cc"

namespace aether {

namespace {

dealii::GridOutFlags::Svg svg_flags() {
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
  return svg;
}

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
  // mesh_symmetric_quarter_pincell(triangulation, {0.44, 0.54}, 0.63, {1, 1, 0});
  std::vector<double> radii{0.4095, 0.4750, 0.54};
  std::vector<int> materials;
  for (int i = 0; i < radii.size() + 1; ++i)
    materials.push_back(i);
  // std::vector<int> materials(radii.size(), 1);
  // materials.push_back(0);
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
  // for (int i = 0; i < radii.size() + 1; ++i)
  //   materials.push_back(i);
  double target = 7.5e-2;
  dealii::Vector<double> measures;
  do {
    measures.reinit(mesh.n_active_cells());
    int c = 0;
    for (auto cell = mesh.begin_active(); cell != mesh.end(); ++cell, ++c)
      measures[c] = std::max(cell->extent_in_direction(1), cell->extent_in_direction(0)); //cell->diameter();
    // dealii::GridRefinement::refine(mesh, measures, target);
    for (auto cell = mesh.begin_active(); cell != mesh.end(); ++cell) {
      // if (cell->refinement_case() == dealii::RefinementCase<2>::no_refinement)
      //   continue;
      if (std::max(cell->extent_in_direction(1),
                   cell->extent_in_direction(0)) < target)
        continue;
      // cell->set_refine_flag(dealii::RefinementCase<2>::no_refinement);
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