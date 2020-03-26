#include "mesh/mesh.h"

namespace aether {

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
  }
  vertices[1+radii.size()*3] = dealii::Point<2>(pitch, 0);
  vertices[2+radii.size()*3] = dealii::Point<2>(0, pitch);
  vertices[3+radii.size()*3] = dealii::Point<2>(pitch, pitch);
  dealii::GridReordering<2> grid_reordering;
  grid_reordering.reorder_cells(cells, true);
  dealii::SubCellData manifolds;
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

void mesh_eighth_pincell(dealii::Triangulation<2> &tria,
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
  vertices[0] = dealii::Point<2>(0, 0);
  cells[0].vertices[0] = 0;
  cells[0].vertices[1] = 1;
  cells[0].vertices[2] = 2;
  cells[0].vertices[3] = 3;
  cells[0].material_id = materials[0];
  for (int i = 0; i < radii.size(); ++i) {
    double diag = radii[i] / std::sqrt(2.0);
    double ring_x = radii[i] * std::cos(dealii::numbers::PI_4 / 2);
    double ring_y = radii[i] * std::sin(dealii::numbers::PI_4 / 2);
    vertices[1+i*3] = dealii::Point<2>(radii[i], 0);
    vertices[2+i*3] = dealii::Point<2>(diag, diag);
    vertices[3+i*3] = dealii::Point<2>(ring_x, ring_y);
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
  }
  vertices[1+radii.size()*3] = dealii::Point<2>(pitch, 0);
  vertices[2+radii.size()*3] = dealii::Point<2>(pitch, pitch);
  vertices[3+radii.size()*3] = dealii::Point<2>(pitch, pitch/2);
  dealii::GridReordering<2> grid_reordering;
  grid_reordering.reorder_cells(cells, true);
  dealii::SubCellData manifolds;
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

void mesh_pincell(dealii::Triangulation<2> &tria,
                  const std::vector<double> &radii,
                  const double &pitch,
                  const std::vector<int> &materials) {
  const dealii::Point<2> center(pitch/2, pitch/2);
  std::vector<dealii::Triangulation<2>> quadrants(4);
  for (int i = 0; i < 4; ++i) {
    mesh_quarter_pincell(quadrants[i], radii, pitch, materials);
    dealii::GridTools::rotate(i*dealii::numbers::PI_2, quadrants[i]);
    dealii::GridTools::shift(center, quadrants[i]);
  }
  dealii::GridGenerator::merge_triangulations(
      {&quadrants[0], &quadrants[1], &quadrants[2], &quadrants[3]}, 
      tria, 1e-12, true);
  tria.set_manifold(1, dealii::SphericalManifold<2>(center));
  dealii::TransfiniteInterpolationManifold<2> trans_manifold;
  trans_manifold.initialize(tria);
  tria.set_manifold(2, trans_manifold);
}

template <int dim>
void set_all_boundaries_reflecting(dealii::Triangulation<dim>& mesh) {
  using Cell = typename dealii::Triangulation<dim>::active_cell_iterator;
  using Face = typename dealii::Triangulation<dim>::active_face_iterator;
  for (Cell cell = mesh.begin_active(); cell != mesh.end(); ++cell) {
    for (int f = 0; f < dealii::GeometryInfo<dim>::faces_per_cell; ++f) {
      Face face = cell->face(f);
      if (face->at_boundary()) {
        face->set_boundary_id(types::reflecting_boundary_id);
      }
    }
  }
}

template void set_all_boundaries_reflecting<1>(dealii::Triangulation<1>&);
template void set_all_boundaries_reflecting<2>(dealii::Triangulation<2>&);
template void set_all_boundaries_reflecting<3>(dealii::Triangulation<3>&);

}  // namespace aether