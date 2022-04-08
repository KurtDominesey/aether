#include <variant>

#include <deal.II/grid/grid_generator.h>

#include "types/types.h"
#include "base/mgxs.h"

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

}  // namespace takeda::lwr


namespace takeda::fbr {

class Benchmark2D1D {
 public:
  // can't make virtual functions templates
  virtual void create_mesh1(dealii::Triangulation<1> &mesh) const = 0;
  virtual void create_mesh2(dealii::Triangulation<2> &mesh) const = 0;
  virtual void create_mesh3(dealii::Triangulation<3> &mesh) const = 0;
  virtual aether::Mgxs create_mgxs() const = 0;
  virtual const std::vector<std::vector<unsigned int>>& get_materials() 
      const = 0;
};

class Fbr : public Benchmark2D1D {
 public:
  static const int num_groups = 4;
  static const int num_layers = 4;
  static const int num_areas = 3;

  enum Material : unsigned int {CORE, ROD, SODIUM, BLANKET_RAD, BLANKET_AX};
  const std::vector<std::vector<unsigned int>> materials{
    {BLANKET_AX, BLANKET_RAD, SODIUM},
    {CORE, BLANKET_RAD, SODIUM},
    {CORE, BLANKET_RAD, ROD},
    {BLANKET_AX, BLANKET_RAD, ROD}
  };

  enum Case : unsigned int {UNRODDED=1, HALF_RODDED};
  const unsigned int case_id = UNRODDED;

  Fbr(Fbr::Case case_id) : case_id(case_id) {};

  template <int dim>
  void create_mesh(dealii::Triangulation<dim> &mesh) const;

  // Benchmark2D1D interface
  void create_mesh1(dealii::Triangulation<1> &mesh) const override {
    create_mesh(mesh);
  }
  void create_mesh2(dealii::Triangulation<2> &mesh) const override {
    create_mesh(mesh);
  }
  void create_mesh3(dealii::Triangulation<3> &mesh) const override {
    create_mesh(mesh);
  }
  const std::vector<std::vector<unsigned int>>& get_materials() const override {
    return materials;
  }
  aether::Mgxs create_mgxs() const override;

 protected:
  const std::array<double, num_layers> layers{20, 75, 130, 150};
  template <int dim>
  unsigned int layer_of(const dealii::Point<dim> &p);
  template <int dim>
  unsigned int area_of(const dealii::Point<dim> &p);
};

template <int dim>
unsigned int Fbr::layer_of(const dealii::Point<dim> &p) {
  AssertThrow(dim != 2, dealii::ExcImpossibleInDim(dim));
  const double z = p[dim == 1 ? 0 : 2];
  for (int i = 0; i < layers.size(); ++i)
    if (z < layers[i])
      return i;
  AssertThrow(false, dealii::ExcInvalidState());
}

template <int dim>
unsigned int Fbr::area_of(const dealii::Point<dim> &p) {
  AssertThrow(dim != 1, dealii::ExcImpossibleInDim(dim));
  if (p[0] > 35 && p[0] < 45 && p[1] < 5)
    return 1;
  bool in_radial_blanket = 
      p[0] > 55 || p[1] > 55 ||  // outside core
      (p[0] > 50 && p[0] < 55 && p[1] > 15) ||  // last row
      (p[1] > 50 && p[1] < 55 && p[0] > 15) ||  // last col
      (p[0] > 45 && p[0] < 50 && p[1] > 30) ||  // second row
      (p[1] > 45 && p[1] < 50 && p[0] > 30) ||  // second col
      (p[0] > 40 && p[0] < 45 && p[1] > 40) ||  // third row
      (p[1] > 40 && p[1] < 45 && p[0] > 40);    // third col
  if (in_radial_blanket)
    return 2;
  else
    return 0;
}

template <int dim>
void Fbr::create_mesh(dealii::Triangulation<dim> &mesh) const {
  if (dim < 1 || dim > 3)
    AssertThrow(false, dealii::ExcImpossibleInDim(dim));
  const double height = 150;
  const double width = 70;
  const double nr = 14;
  const double nz = 30;
  std::vector<unsigned int> nd;
  dealii::Point<dim> p0, p1;
  switch (dim) {
    case 1:
      nd = {nz};
      p1 = dealii::Point<dim>(height);
      break;
    case 2:
      nd = {nr, nr};
      p1 = dealii::Point<dim>(width, width);
      break;
    case 3:
      nd = {nr, nr, nz};
      p1 = dealii::Point<dim>(width, width, height);
      break;
  }
  dealii::GridGenerator::subdivided_hyper_rectangle(mesh, nd, p0, p1, true);
  for (auto &cell : mesh.active_cell_iterators()) {
    const dealii::Point<dim> &p = cell->center();
    unsigned int matl = 0;
    switch (dim) {
      case 1: matl = layer_of(p); break;
      case 2: matl = area_of(p); break;
      case 3: matl = materials[layer_of(p)][area_of(p)];
    }
    cell->set_material_id(matl);
    for (int f = 0; f < dealii::GeometryInfo<dim>::faces_per_cell; ++f) {
      // replace interior x-y (0, 2) with reflecting
      // replace exterior x-y (1, 3) and z (4, 5) with vacuum (0)
      dealii::types::boundary_id b = cell->face(f)->boundary_id();
      if (b == 0 || b == 2)
        cell->face(f)->set_boundary_id(aether::types::reflecting_boundary_id);
      else if (b == 1 || b == 3 || b == 4 || b == 5)
        cell->face(f)->set_boundary_id(0);
      else
        AssertThrow(b == dealii::numbers::internal_face_boundary_id, 
                    dealii::ExcInvalidState());
    }
  }
}

aether::Mgxs Fbr::create_mgxs() const {

}

}