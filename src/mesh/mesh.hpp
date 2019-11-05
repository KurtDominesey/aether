#ifndef AETHER_MESH_H_
#define AETHER_MESH_H_

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_reordering.h>
#include <deal.II/grid/manifold_lib.h>

void mesh_quarter_pincell(dealii::Triangulation<2> &tria,
                          const std::vector<double> radii,
                          const double pitch,
                          const std::vector<int> materials) {
  AssertDimension(materials.size(), radii.size()+1);
  if (!radii.empty())
    Assert(radii.back() < pitch, dealii::ExcInvalidState());
  for (int i = 1; i < radii.size(); ++i)
    Assert(radii[i-1] < radii[i], dealii::ExcInvalidState());
  std::vector<dealii::Point<2>> vertices(3*radii.size()+4);
  std::vector<dealii::CellData<2>> cells(2*radii.size()+1);
  std::vector<dealii::CellData<1>> subcells(2*radii.size()+(radii.size()-1));
  vertices[0] = dealii::Point<2>(0, 0);
  cells[0].vertices[0] = 0;
  cells[0].vertices[1] = 1;
  cells[0].vertices[2] = 2;
  cells[0].vertices[3] = 3;
  cells[0].material_id = materials[0];
  for (int i = 0; i < radii.size(); ++i) {
    double diag = radii[i] / (std::sqrt(2.0));
    vertices[1+i*3] = dealii::Point<2>(radii[i], 0);
    vertices[2+i*3] = dealii::Point<2>(0, radii[i]);
    vertices[3+i*3] = dealii::Point<2>(diag, diag);
    cells[1+i*2].vertices[0] = 2 + i * 3;
    cells[1+i*2].vertices[1] = 3 + i * 3;
    cells[1+i*2].vertices[2] = 2 + (i + 1) * 3;
    cells[1+i*2].vertices[3] = 3 + (i + 1) * 3;
    cells[2+i*2].vertices[0] = 3 + i * 3;
    cells[2+i*2].vertices[1] = 1 + i * 3;
    cells[2+i*2].vertices[2] = 3 + (i + 1) * 3;
    cells[2+i*2].vertices[3] = 1 + (i + 1) * 3;
    cells[1+i*2].material_id = materials[i+1];
    cells[2+i*2].material_id = materials[i+1];
    subcells[i*2].vertices[0] = 2 + i * 3;
    subcells[i*2].vertices[1] = 3 + i * 3;
    subcells[i*2+1].vertices[0] = 3 + i * 3;
    subcells[i*2+1].vertices[1] = 1 + i * 3;
    subcells[i*2].manifold_id = 1;
    subcells[i*2+1].manifold_id = 1;
    subcells[i*2].boundary_id = dealii::numbers::internal_face_boundary_id;
    subcells[i*2+1].boundary_id = dealii::numbers::internal_face_boundary_id;
  }
  vertices[1+radii.size()*3] = dealii::Point<2>(pitch, 0);
  vertices[2+radii.size()*3] = dealii::Point<2>(0, pitch);
  vertices[3+radii.size()*3] = dealii::Point<2>(pitch, pitch);
  dealii::SubCellData manifolds;
  // manifolds.boundary_lines = subcells;
  dealii::GridReordering<2> grid_reordering;
  grid_reordering.reorder_cells(cells, true);
  tria.create_triangulation(vertices, cells, manifolds);
  tria.set_all_manifold_ids(1);
  tria.set_all_manifold_ids_on_boundary(dealii::numbers::flat_manifold_id);
  tria.begin()->set_manifold_id(2);
  int i = 0;
  for (auto cell = tria.last(); i < 2; ++i, --cell)
    cell->set_manifold_id(2);
  tria.set_manifold(1, dealii::SphericalManifold<2>());
  dealii::TransfiniteInterpolationManifold<2> trans_manifold;
  trans_manifold.initialize(tria);
  tria.set_manifold(2, trans_manifold);
}

#endif  // AETHER_MESH_H_