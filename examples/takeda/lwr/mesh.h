#include <deal.II/grid/grid_generator.h>

#include "types/types.h"
#include "mesh/mesh.h"
#include "base/mgxs.h"

class Benchmark2D1D {
 public:
  // can't make virtual functions templates
  virtual void create_mesh1(dealii::Triangulation<1> &mesh) const = 0;
  virtual void create_mesh2(dealii::Triangulation<2> &mesh) const = 0;
  virtual void create_mesh3(dealii::Triangulation<3> &mesh) const = 0;
  virtual std::unique_ptr<dealii::Function<1>> 
      create_source1(const double intensity) const = 0;
  virtual std::unique_ptr<dealii::Function<2>> 
      create_source2(const double intensity) const = 0;
  virtual std::unique_ptr<dealii::Function<3>> 
      create_source3(const double intensity) const = 0;
  virtual aether::Mgxs create_mgxs() const = 0;
  virtual const std::vector<std::vector<int>>& get_materials() 
      const = 0;
  virtual std::string to_string() const = 0;
};

/**
 * Solution of the Helmholtz equation on a cuboid.
 */
template <int dim>
class BucklingCuboid : public dealii::Function<dim> {
 public:
  BucklingCuboid(const double intensity,
                 const dealii::Point<dim> &center, 
                 const dealii::Point<dim> &corner)
                 : intensity(intensity), center(center) {
    dist = corner - center;
    for (int d = 0; d < dim; ++d)
      freq[d] = dealii::numbers::PI_2 / dist[d];
  };
  double value(const dealii::Point<dim> &p, unsigned int=0) const override {
    double v = intensity;
    dist = p - center;
    for (int i = 0; i < dim; ++i)
      v *= std::cos(freq[i]*dist[i]);
    return v;
   }

 protected:
  const double intensity;
  const dealii::Point<dim> center;
  // Calling operator- on a point returns a tensor for some reason.
  // I assume the argument/return types were mistakenly swapped.
  mutable dealii::Tensor<1, dim> dist;
  dealii::Point<dim> freq;
};

/**
 * Solution of the Helmholtz equation on a cylinder.
 */
template <int dim>
class BucklingCyl : public dealii::Function<dim> {
 public:
  BucklingCyl(const double intensity,
              const dealii::Point<dim> &center, 
              const double radius, 
              const double height=0) : 
              intensity(intensity), 
              center(center), 
              root_radius(root/radius), 
              pi_height(dealii::numbers::PI/height) {
    
  }
  double value(const dealii::Point<dim> &p, unsigned int=0) const override {
    const double r = p.distance(center);
    double v = intensity;
    if (dim != 2) {
      const int d = dim == 1 ? 0 : 2;
      v *= std::cos(pi_height*(p[d]-center[d]));
    }
    if (dim >= 2)
      v *= std::cyl_bessel_j(0, root_radius*r);
    return std::max(v, 0.);
  }

 protected:
  const double root = 2.405;
  const double intensity, root_radius, pi_height;
  const dealii::Point<dim> center;
};

namespace takeda {

enum Model : int {
  LWR = 1,
  FBR_SMALL,
  FBR_HET_Z,
  FBR_HEX_Z
};

class Lwr : public Benchmark2D1D {
 public:
  enum Material : unsigned int {CORE, REFL, ROD};
  const std::vector<std::vector<int>> materials{
    {CORE, REFL, ROD},
    {REFL, REFL, ROD}
  };

  const bool is_rodded;
  Lwr(bool is_rodded) : is_rodded(is_rodded) {};

  template <int dim>
  void create_mesh(dealii::Triangulation<dim> &mesh) const;

  template <int dim>
  std::unique_ptr<dealii::Function<dim>> 
  create_source(const double intensity) const;

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
  std::unique_ptr<dealii::Function<1>>
  create_source1(const double intensity) const override {
    return create_source<1>(intensity);
  }
  std::unique_ptr<dealii::Function<2>> 
  create_source2(const double intensity) const override {
    return create_source<2>(intensity);
  }
  std::unique_ptr<dealii::Function<3>> 
  create_source3(const double intensity) const override {
    return create_source<3>(intensity);
  }
  const std::vector<std::vector<int>>& get_materials() const override {
    return materials;
  }
  aether::Mgxs create_mgxs() const override;
  std::string to_string() const override;
};

template <int dim>
void Lwr::create_mesh(dealii::Triangulation<dim> &mesh) const {
  if (dim < 1 || dim > 3)
    AssertThrow(false, dealii::ExcImpossibleInDim(dim));
  // Using subdivided_hyper_cube would avoid this boilerplate.
  // But in deal.II 9.1.1, only the hyper-rectangle supports colorization.
  // This is fixed in 9.2.0, so this can be refactored when we update.
  std::vector<unsigned int> num_el(dim, 25);
  dealii::Point<dim> p0, p1;
  for (int i = 0; i < dim; ++i)
    p1[i] = 25;
  dealii::GridGenerator::subdivided_hyper_rectangle(mesh, num_el, p0, p1, true);
  auto matl_of = [](const dealii::Point<dim> &p) -> unsigned int {
    switch (dim) {
      case 1: return p[0] < 15 ? 0 : 1;
      case 2:
        if (p[0] < 15 && p[1] < 15)
          return CORE;
        else if (p[0] < 20 && p[1] < 5)
          return ROD;
        else
          return REFL;
      case 3:
        if (p[0] < 15 && p[1] < 15)
          return p[2] < 15 ? CORE : REFL;
        else if (p[0] < 20 && p[1] < 5)
          return ROD;
        else
          return REFL;
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

template <int dim>
std::unique_ptr<dealii::Function<dim>> Lwr::create_source(
    const double intensity) const {
  dealii::Point<dim> center, corner;
  for (int d = 0; d < dim; ++d)
    corner[d] = 15;
  return std::make_unique<BucklingCuboid<dim>>(intensity, center, corner);
}

aether::Mgxs Lwr::create_mgxs() const {
  const int num_groups = 2;
  aether::Mgxs mgxs(2/*groups*/, 3/*materials*/, 1/*legendre moment*/);
  mgxs.group_structure = {1e-5, 6.8256e-1, 1e7};
  for (int g = 0; g < num_groups; ++g) {
    mgxs.group_widths[num_groups-1-g] = 
        std::log(mgxs.group_structure[g+1]/mgxs.group_structure[g]);
  }
  mgxs.total[0] = {2.23775e-1, 2.50367e-1, is_rodded ? 8.52325e-2 : 1.28407e-2};
  mgxs.total[1] = {1.03864,    1.64482,    is_rodded ? 2.17460e-1 : 1.20676e-2};
  mgxs.scatter[0][0] = {1.92423e-1, 1.93446e-1, is_rodded ? 6.77241e-2 : 1.27700e-2};
  mgxs.scatter[0][1] = {0.0, 0.0, 0.0};
  mgxs.scatter[1][0] = {2.28253e-2, 5.65042e-2, is_rodded ? 6.45461e-5 : 2.40997e-5};
  mgxs.scatter[1][1] = {8.80439e-1, 1.62452,    is_rodded ? 3.52358e-2 : 1.07387e-2};
  mgxs.nu_fission[0] = {9.09319e-3, 0.0, 0.0};
  mgxs.nu_fission[1] = {2.90183e-1, 0.0, 0.0};
  mgxs.chi[0] = {1.0, 0.0, 0.0};
  mgxs.chi[1] = {0.0, 0.0, 0.0};
  return mgxs;
}

std::string Lwr::to_string() const {
  return std::string("LwrRod") + (is_rodded ? "Y" : "N");
}


// Used for FBR Models 2 & 3
aether::Mgxs create_mgxs_fbr() {
  // set up materials
  const int num_groups = 4;
  // core, radial blanket, axial blanket, control rod (cr), cr position (crp)
  const int num_materials = 5;
  std::vector<double> chi = {0.583319, 0.405450, 0.011231, 0};
  std::vector<std::vector<double>> total = {
      {1.14568e-1, 2.05177e-1, 3.29381e-1, 3.89810e-1},  // core
      {1.19648e-1, 2.42195e-1, 3.56476e-1, 3.79433e-1},  // radial blanket
      {1.16493e-1, 2.20521e-1, 3.44544e-1, 3.88356e-1},  // axial blanket
      {1.84333e-1, 3.66121e-1, 6.15527e-1, 1.09486e+0},  // control rod (cr)
      {6.58979e-2, 1.09810e-1, 1.86765e-1, 2.09933e-1}   // cr position (crp)
  };
  std::vector<std::vector<double>> nu_fission = {
      {2.06063e-2, 6.10571e-3, 6.91403e-3, 2.60689e-2},
      {1.89496e-2, 1.75265e-4, 2.06978e-4, 1.13451e-3},
      {1.31770e-2, 1.26026e-4, 1.52380e-4, 7.87302e-4},
      {0,          0,          0,          0},
      {0,          0,          0,          0}
  };
  std::vector<std::vector<std::vector<double>>> scatter(num_materials,
      std::vector<std::vector<double>>(num_groups, 
        std::vector<double>(num_groups)));
  scatter[0] = {  // core
      {7.04326e-2, 0,          0,          0},
      {3.47967e-2, 1.95443e-1, 0,          0},
      {1.88282e-3, 6.20863e-3, 3.20586e-1, 0},
      {0,          7.07208e-7, 9.92975e-4, 3.62360e-1}
  };
  scatter[1] = {  // radial blanket
      {6.91158e-2, 0,          0,          0},
      {4.04132e-2, 2.30626e-1, 0,          0},
      {2.68621e-3, 9.57027e-3, 3.48414e-1, 0},
      {0,          1.99571e-7, 1.27195e-3, 3.63631e-1}
  };
  scatter[2] = {  // axial blanket
      {7.16044e-2, 0,          0,          0},
      {3.73170e-2, 2.10436e-1, 0,          0},
      {2.21707e-3, 8.59855e-3, 3.37506e-1, 0},
      {0,          6.68299e-7, 1.68530e-3, 3.74886e-1}
  };
  scatter[3] = {  // control rod
      {1.34373e-1, 0,          0,          0},
      {4.37775e-2, 3.18582e-1, 0,          0},
      {2.06054e-4, 2.98432e-2, 5.19591e-1, 0},
      {0,          8.71188e-7, 7.66209e-3, 6.18265e-1}
  };
  scatter[4] = {  // control rod position
      {4.74407e-2, 0,          0,          0},
      {1.76894e-2, 1.06142e-1, 0,          0},
      {4.57012e-4, 3.55466e-3, 1.85304e-1, 0},
      {0,          1.77599e-7, 1.01280e-3, 2.08858e-1}
  };
  aether::Mgxs mgxs(num_groups, num_materials, 1);
  for (int j = 0; j < num_materials; ++j) {
    for (int g = 0; g < num_groups; ++g) {
      mgxs.total[g][j] = total[j][g];
      mgxs.nu_fission[g][j] = nu_fission[j][g];
      mgxs.chi[g][j] = j <= 2 ? chi[g] : 0;
      for (int gp = 0; gp < num_groups; ++gp)
        mgxs.scatter[g][gp][j] = scatter[j][g][gp];
    }
  }
  return mgxs;
}


class Fbr : public Benchmark2D1D {
 public:
  const bool is_rodded;  // otherwise, is half-rodded
  enum Material : int {CORE, BLANKET_RAD, BLANKET_AX, ROD, SODIUM};
  const std::vector<std::vector<int>> materials{
    {BLANKET_AX, BLANKET_RAD, is_rodded ? ROD : SODIUM},
    {CORE, BLANKET_RAD, is_rodded ? ROD : SODIUM},
    {CORE, BLANKET_RAD, ROD},
    {BLANKET_AX, BLANKET_RAD, ROD}
  };

  Fbr(bool is_rodded) : is_rodded(is_rodded) {};

  template <int dim>
  void create_mesh(dealii::Triangulation<dim> &mesh) const;

  template <int dim>
  std::unique_ptr<dealii::Function<dim>> 
  create_source(const double intensity) const;

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
  std::unique_ptr<dealii::Function<1>>
  create_source1(const double intensity) const override {
    return create_source<1>(intensity);
  }
  std::unique_ptr<dealii::Function<2>> 
  create_source2(const double intensity) const override {
    return create_source<2>(intensity);
  }
  std::unique_ptr<dealii::Function<3>> 
  create_source3(const double intensity) const override {
    return create_source<3>(intensity);
  }
  const std::vector<std::vector<int>>& get_materials() const override {
    return materials;
  }
  aether::Mgxs create_mgxs() const override;
  std::string to_string() const override;

 protected:
  const double radius = std::sqrt(2) * 40.;
  const double height = 150;
  const std::array<double, 4> layers{20, 75, 130, 150};
  template <int dim>
  unsigned int layer_of(const dealii::Point<dim> &p) const;
  template <int dim>
  unsigned int area_of(const dealii::Point<dim> &p) const;
};

template <int dim>
unsigned int Fbr::layer_of(const dealii::Point<dim> &p) const {
  AssertThrow(dim != 2, dealii::ExcImpossibleInDim(dim));
  const double z = p[dim == 1 ? 0 : 2];
  for (int i = 0; i < layers.size(); ++i)
    if (z < layers[i])
      return i;
  AssertThrow(false, dealii::ExcInvalidState());
}

template <int dim>
unsigned int Fbr::area_of(const dealii::Point<dim> &p) const {
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

template <int dim> 
std::unique_ptr<dealii::Function<dim>> Fbr::create_source(
    const double intensity) const {
  dealii::Point<dim> center;
  if (dim != 2)
    center[dim == 1 ? 0 : 2] = height / 2.;
  return std::make_unique<BucklingCyl<dim>>(
    intensity, center, radius, 110
  );
} 

aether::Mgxs Fbr::create_mgxs() const {
  return create_mgxs_fbr();
}

std::string Fbr::to_string() const {
  return std::string("FbrRod") + (is_rodded ? "H" : "N");
}


class PinC5G7 : public Benchmark2D1D {
 public:
  enum Material : int {FUEL, WATER};
  std::vector<std::vector<int>> materials{
    {FUEL,  WATER},
    {WATER, WATER}
  };
  const std::string fuel;  // name in c5g7.h5, e.g. "uo2" or "mox43"

  const double height = 64.26;
  const double height_pin = 42.84;
  const unsigned int nz = 51;  // each layer is 0.56 cm thick (before refinement)
  const int num_refined = 1;

  PinC5G7(const std::string fuel) : fuel(fuel) {};

  template <int dim>
  std::unique_ptr<dealii::Function<dim>>
  create_source(const double intensity) const;

  void create_mesh1_coarse(dealii::Triangulation<1> &mesh) const;
  void create_mesh2_coarse(dealii::Triangulation<2> &mesh) const;

  // Benchmark2D1D interface
  void create_mesh1(dealii::Triangulation<1> &mesh) const override;
  void create_mesh2(dealii::Triangulation<2> &mesh) const override;
  void create_mesh3(dealii::Triangulation<3> &mesh) const override;
  std::unique_ptr<dealii::Function<1>>
  create_source1(const double intensity) const override {
    return create_source<1>(intensity);
  }
  std::unique_ptr<dealii::Function<2>> 
  create_source2(const double intensity) const override {
    return create_source<2>(intensity);
  }
  std::unique_ptr<dealii::Function<3>> 
  create_source3(const double intensity) const override {
    return create_source<3>(intensity);
  }
  const std::vector<std::vector<int>>& get_materials() const override {
    return materials;
  }
  aether::Mgxs create_mgxs() const override;
  std::string to_string() const override;
};

void PinC5G7::create_mesh1(dealii::Triangulation<1> &mesh) const {
  create_mesh1_coarse(mesh);
  mesh.refine_global(num_refined);
}

void PinC5G7::create_mesh1_coarse(dealii::Triangulation<1> &mesh) const {
  dealii::Point<1> p0, p1;
  p1[0] = height;
  std::vector<unsigned int> nnz = {nz};
  dealii::GridGenerator::subdivided_hyper_rectangle(mesh, nnz, p0, p1);
  for (auto &cell : mesh.active_cell_iterators()) {
    cell->set_material_id(cell->center()[0] > height_pin);
    for (int f = 0; f < dealii::GeometryInfo<1>::faces_per_cell; ++f) {
      auto b = cell->face(f)->boundary_id();
      if (b == 0)
        cell->face(f)->set_boundary_id(aether::types::reflecting_boundary_id);
      else if (b == 1)
        cell->face(f)->set_boundary_id(0);
      else
        AssertThrow(b == dealii::numbers::internal_face_boundary_id,
                    dealii::ExcInvalidState());
    }
  }
}

void PinC5G7::create_mesh2(dealii::Triangulation<2> &mesh) const {
  create_mesh2_coarse(mesh);
  aether::set_all_boundaries_reflecting(mesh);
  mesh.refine_global(num_refined);
}

void PinC5G7::create_mesh2_coarse(dealii::Triangulation<2> &mesh) const {
  aether::mesh_symmetric_quarter_pincell(
      mesh, {0.54}, 0.63, materials[0], 1, 2, true);
}

void PinC5G7::create_mesh3(dealii::Triangulation<3> &mesh) const {
  dealii::Triangulation<2> mesh_rad;
  create_mesh2_coarse(mesh_rad);
  const std::vector<dealii::types::manifold_id> priorities =
      {2, 1, dealii::numbers::flat_manifold_id};
  dealii::GridGenerator::extrude_triangulation(
      mesh_rad, nz+1, height, mesh, true, priorities);
  for (auto &cell : mesh.active_cell_iterators()) {
    if (cell->center()[2] > height_pin)
      cell->set_material_id(WATER);
    for (int f = 0; f < dealii::GeometryInfo<3>::faces_per_cell; ++f) {
      const auto b = cell->face(f)->boundary_id();
      if (b == 2)
        cell->face(f)->set_boundary_id(0);
      else if (b == 0 || b == 1)
        cell->face(f)->set_boundary_id(aether::types::reflecting_boundary_id);
      else
        AssertThrow(b == dealii::numbers::internal_face_boundary_id,
                    dealii::ExcInvalidState());
    }
  }
  dealii::CylindricalManifold<3> mf_cyl(2);
  dealii::TransfiniteInterpolationManifold<3> mf_trans;
  mf_trans.initialize(mesh);
  mesh.set_manifold(1, mf_trans);
  mesh.set_manifold(2, mf_cyl);
  mesh.refine_global(num_refined);
}

template <int dim>
std::unique_ptr<dealii::Function<dim>> PinC5G7::create_source(
    const double intensity) const {
  return std::make_unique<dealii::Functions::ConstantFunction<dim>>(intensity);
}

aether::Mgxs PinC5G7::create_mgxs() const {
  aether::Mgxs mgxs(7, 2, 1);
  read_mgxs(mgxs, "../../c5g7-uo2/c5g7.h5", "294K", {fuel, "water"}, true);
  return mgxs;
}

std::string PinC5G7::to_string() const {
  std::string name = "Pin";
  if (fuel == "uo2")
    name += "UO2";
  else if (fuel == "mox43")
    name += "Mox43";
  else 
    AssertThrow(false, dealii::ExcNotImplemented());
  return name;
}

}