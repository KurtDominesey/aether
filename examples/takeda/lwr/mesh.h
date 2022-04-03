#include <deal.II/grid/grid_generator.h>

#include "types/types.h"

namespace takeda::lwr {

const double LENGTH = 25.0;

template <int dim>
void create_mesh(dealii::Triangulation<dim> &mesh) {
  if (dim < 1 || dim > 3)
    AssertThrow(false, dealii::ExcImpossibleInDim(dim));
  // Using subdivided_hyper_cube would avoid this boilerplate.
  // But in deal.II 9.1.1, only the hyper-rectangle supports colorization.
  // This is fixed in 9.2.0, so this can be refactored when we update.
  std::vector<unsigned int> num_el(dim, 25);
  dealii::Point<dim> p0, p1;
  for (int i = 0; i < dim; ++i)
    p1[i] = LENGTH;
  dealii::GridGenerator::subdivided_hyper_rectangle(mesh, num_el, p0, p1, true);
  auto matl_of = [](const dealii::Point<dim> &p) -> unsigned int {
    switch (dim) {
      case 1: return p[0] < 15 ? 0 : 1;
      case 2:
        if (p[0] < 15 && p[1] < 15)
          return 0;
        else if (p[0] < 20 && p[1] < 5)
          return 2;
        else
          return 1;
      case 3:
        if (p[0] < 15 && p[1] < 15)
          return p[2] < 15 ? 0 : 1;
        else if (p[0] < 20 && p[1] < 5)
          return 2;
        else
          return 1;
    }
  };
  for (auto &cell : mesh.active_cell_iterators()) {
    cell->set_material_id(matl_of(cell->center()));
    for (int f = 0; f < dealii::GeometryInfo<dim>::faces_per_cell; ++f) {
      // set interior (0, 2, 4) to reflecting
      // set exterior (1, 3, 5) with vacuum (0)
      dealii::types::boundary_id b = cell->face(f)->boundary_id();
      if (b == 0 || b == 2 || b == 4)
        cell->face(f)->set_boundary_id(aether::types::reflecting_boundary_id);
      else if (b == 1 || b == 3 || b == 5)
        cell->face(f)->set_boundary_id(0);
      else
        AssertThrow(b == dealii::numbers::internal_face_boundary_id, 
                    dealii::ExcInvalidState());
    }
  }
}

}