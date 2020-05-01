#include "mesh/mesh.h"

namespace aether {

void mesh_quarter_pincell(dealii::Triangulation<2> &tria,
                          const std::vector<double> radii,
                          const double pitch,
                          const std::vector<int> materials,
                          const int trans_mani_id,
                          const int sph_mani_id) {
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
  tria.set_all_manifold_ids(sph_mani_id);
  tria.set_all_manifold_ids_on_boundary(dealii::numbers::flat_manifold_id);
  tria.begin()->set_manifold_id(trans_mani_id);
  int i = 0;
  for (auto cell = tria.last(); i < 2; ++i, --cell)
    cell->set_manifold_id(trans_mani_id);
  tria.set_manifold(1, dealii::SphericalManifold<2>());
  dealii::TransfiniteInterpolationManifold<2> trans_manifold;
  trans_manifold.initialize(tria);
  tria.set_manifold(trans_mani_id, trans_manifold);
}

void mesh_eighth_pincell(dealii::Triangulation<2> &tria,
                          std::vector<double> radii,
                          const double pitch,
                          std::vector<int> materials,
                          const int trans_mani_id,
                          const int sph_mani_id) {
  AssertDimension(materials.size(), radii.size()+1);
  if (!radii.empty())
    Assert(radii.back() < pitch, dealii::ExcInvalidState());
  for (int i = 1; i < radii.size(); ++i)
    Assert(radii[i-1] < radii[i], dealii::ExcInvalidState());
  // double r0 = radii[0] / (1 + std::sqrt(0.5));
  double a = dealii::numbers::PI_4 / 2;
  double r0 = std::sqrt(dealii::numbers::PI * std::sin(a) * std::cos(a)
                        / 3.0) * radii[0];
  radii.insert(radii.begin(), r0);
  materials.insert(materials.begin(), materials[0]);
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
    if (i == 0) {
      vertices[1+i*3] = dealii::Point<2>(ring_x, 0);
      vertices[2+i*3] = dealii::Point<2>(ring_x/std::sqrt(2.0),
                                         ring_x/std::sqrt(2.0));
    }
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
  double wall_y = pitch * std::tan(dealii::numbers::PI_4 / 2);
  vertices[1+radii.size()*3] = dealii::Point<2>(pitch, 0);
  vertices[2+radii.size()*3] = dealii::Point<2>(pitch, pitch);
  vertices[3+radii.size()*3] = dealii::Point<2>(pitch, wall_y);
  dealii::GridReordering<2> grid_reordering;
  grid_reordering.reorder_cells(cells, true);
  dealii::SubCellData manifolds;
  tria.create_triangulation(vertices, cells, manifolds);
  tria.set_all_manifold_ids(sph_mani_id);
  tria.set_all_manifold_ids_on_boundary(dealii::numbers::flat_manifold_id);
  int i = 0;
  for (auto cell = tria.last(); i < 2; ++i, --cell)
    cell->set_manifold_id(trans_mani_id);
  i = 0;
  for (auto cell = ++tria.begin(); i < 2; ++i, ++cell)
    cell->set_manifold_id(trans_mani_id);
  tria.begin()->set_all_manifold_ids(dealii::numbers::flat_manifold_id);
  tria.set_manifold(sph_mani_id, dealii::SphericalManifold<2>());
  dealii::TransfiniteInterpolationManifold<2> trans_manifold;
  trans_manifold.initialize(tria);
  tria.set_manifold(trans_mani_id, trans_manifold);
}

void mesh_eighth_pincell_ul(dealii::Triangulation<2> &tria,
                            std::vector<double> radii,
                            const double pitch,
                            std::vector<int> materials,
                            const int trans_mani_id,
                            const int sph_mani_id) {
  AssertDimension(materials.size(), radii.size()+1);
  if (!radii.empty())
    Assert(radii.back() < pitch, dealii::ExcInvalidState());
  for (int i = 1; i < radii.size(); ++i)
    Assert(radii[i-1] < radii[i], dealii::ExcInvalidState());
  // r0**2 = 0.5 * (r1 - r0)**2
  // r0 = sqrt(0.5) (r1 - r0)
  // (1 + sqrt(0.5)) r0 = r1
  // r0 = r1 / (1 + sqrt(0.5))
  // double r0 = radii[0] / (1 + std::sqrt(0.5));
  double a = dealii::numbers::PI_4 / 2;
  double r0 = std::sqrt(dealii::numbers::PI * std::sin(a) * std::cos(a)
                        / 3.0) * radii[0];
  radii.insert(radii.begin(), r0);
  materials.insert(materials.begin(), materials[0]);
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
    double ring_x = radii[i] * std::cos(dealii::numbers::PI_4 * 1.5);
    double ring_y = radii[i] * std::sin(dealii::numbers::PI_4 * 1.5);
    vertices[1+i*3] = dealii::Point<2>(diag, diag);
    vertices[2+i*3] = dealii::Point<2>(0, radii[i]);
    vertices[3+i*3] = dealii::Point<2>(ring_x, ring_y);
    if (i == 0) {
      vertices[1+i*3] = dealii::Point<2>(ring_y/std::sqrt(2.0),
                                         ring_y/std::sqrt(2.0));
      vertices[2+i*3] = dealii::Point<2>(0, ring_y);
    }
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
  double wall_x = pitch / std::tan(dealii::numbers::PI_4 * 1.5);
  vertices[1+radii.size()*3] = dealii::Point<2>(pitch, pitch);
  vertices[2+radii.size()*3] = dealii::Point<2>(0, pitch);
  vertices[3+radii.size()*3] = dealii::Point<2>(wall_x, pitch);
  dealii::GridReordering<2> grid_reordering;
  grid_reordering.reorder_cells(cells, true);
  dealii::SubCellData manifolds;
  tria.create_triangulation(vertices, cells, manifolds);
  tria.set_all_manifold_ids(sph_mani_id);
  tria.set_all_manifold_ids_on_boundary(dealii::numbers::flat_manifold_id);
  int i = 0;
  for (auto cell = tria.last(); i < 2; ++i, --cell)
    cell->set_manifold_id(trans_mani_id);
  i = 0;
  for (auto cell = ++tria.begin(); i < 2; ++i, ++cell)
    cell->set_manifold_id(trans_mani_id);
  tria.begin()->set_all_manifold_ids(dealii::numbers::flat_manifold_id);
  tria.set_manifold(sph_mani_id, dealii::SphericalManifold<2>());
  dealii::TransfiniteInterpolationManifold<2> trans_manifold;
  trans_manifold.initialize(tria);
  tria.set_manifold(trans_mani_id, trans_manifold);
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

void mesh_symmetric_quarter_pincell(dealii::Triangulation<2> &tria,
                                    const std::vector<double> &radii,
                                    const double &pitch,
                                    const std::vector<int> &materials,
                                    const int trans_mani_id,
                                    const int sph_mani_id) {
  dealii::Triangulation<2> octant_ul;
  dealii::Triangulation<2> octant_lr;
  mesh_eighth_pincell(octant_lr, radii, pitch, materials, 
                      trans_mani_id, sph_mani_id);
  mesh_eighth_pincell_ul(octant_ul, radii, pitch, materials,
                         trans_mani_id, sph_mani_id);
  dealii::GridGenerator::merge_triangulations(
      octant_ul, octant_lr, tria, 1e-12, true);
  tria.set_manifold(sph_mani_id, dealii::SphericalManifold<2>());
  dealii::TransfiniteInterpolationManifold<2> trans_manifold;
  trans_manifold.initialize(tria);
  tria.set_manifold(trans_mani_id, trans_manifold);
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

void mesh_mox_assembly(dealii::Triangulation<2> &mesh) {
  // std::vector<int> materials_f = {2, 0, 3, 0, 4, 1};
  std::vector<int> materials_f = {2, 3, 4, 1};
  std::vector<int> materials_g = {1, 4, 1};
  // std::vector<double> radii_f = {0.4095, 0.4180, 0.4750, 0.4850, 0.54};
  std::vector<double> radii_f = {0.4095, 0.4750, 0.54};
  std::vector<double> radii_g = {0.34, 0.54};
  const double pitch = 1.26;
  std::vector<std::vector<std::string>> mox_assm = {
  // 09   10   11   12   13   14   15   16   17
    {"l", "l", "l", "l", "l", "l", "l", "l", "l"}, // 1
    {"m", "m", "m", "m", "m", "m", "m", "m"}, // 2
    {"g", "m", "m", "g", "m", "m", "m"}, // 3
    {"h", "h", "h", "h", "m", "g"}, // 4
    {"h", "h", "h", "h", "h"}, // 5
    {"g", "h", "h", "g"}, // 6
    {"h", "h", "h"}, // 7
    {"h", "h"}, // 8
    {"g"}  // 9
  };
  for (int i = 0, ij = 0; i < mox_assm.size(); ++i) {
    for (int j = 0; j < mox_assm[i].size(); ++j, ++ij) {
      std::vector<int> materials;
      std::vector<double> radii;
      if (mox_assm[i][j] == "g") {
        materials = materials_g;
        radii = radii_g;
      } else {
        materials = materials_f;
        radii = radii_f;
      }
      bool dcut = j == mox_assm[i].size() - 1;
      bool vcut = j == 0;
      dealii::Point<2> center(j*pitch, i*pitch);
      std::cout << j*pitch << ", " << i*pitch << "\n";
      std::vector<dealii::Triangulation<2>> octants(8);
      for (int r = 0; r < 8; ++r) {
        if (dcut && (r < 3 || r == 7))
          continue;
        if (vcut && (r > 1 && r < 6))
          continue;
        if (r % 2)
          mesh_eighth_pincell_ul(octants[r], radii, pitch/2, materials, 1, ij+2);
        else
          mesh_eighth_pincell(octants[r], radii, pitch/2, materials, 1, ij+2);
        int incr = std::floor(r/2);
        dealii::GridTools::rotate(incr*dealii::numbers::PI_2, octants[r]);
        dealii::GridTools::shift(center, octants[r]);
        dealii::GridGenerator::merge_triangulations(
            octants[r], mesh, mesh, 1e-6, true);
      }
    }
  }
  for (int i = 0, ij = 0; i < mox_assm.size(); ++i) {
    for (int j = 0; j < mox_assm[i].size(); ++j, ++ij) {
      dealii::Point<2> center(j*pitch, i*pitch);
      mesh.set_manifold(ij+2, dealii::SphericalManifold<2>(center));
    }
  }
  dealii::TransfiniteInterpolationManifold<2> trans_manifold;
  trans_manifold.initialize(mesh);
  mesh.set_manifold(1, trans_manifold);
}

}  // namespace aether