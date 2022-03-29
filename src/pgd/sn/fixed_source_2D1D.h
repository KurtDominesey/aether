#ifndef AETHER_PGD_SN_FIXED_SOURCE_2D1D_H_
#define AETHER_PGD_SN_FIXED_SOURCE_2D1D_H_

#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_richardson.h>
#include <deal.II/lac/block_vector.h>

#include "base/mgxs.h"
#include "sn/fixed_source.h"
#include "sn/quadrature.h"
#include "pgd/sn/transport_block.h"
#include "pgd/sn/inner_products.h"
#include "pgd/sn/linear_interface.h"

namespace aether::pgd::sn {

struct Products {
  Products(int groups, int ords, int dofs)
  : psi(groups, ords*dofs), phi(groups, dofs), streamed(groups, ords*dofs) {};
  dealii::BlockVector<double> psi;
  dealii::BlockVector<double> phi;
  dealii::BlockVector<double> streamed;
};

template <int dim, int qdim, int zonesND, int groupsND>
class FixedSource2D1D {
 public:
  FixedSource2D1D(const aether::sn::FixedSource<dim, qdim> &fixed_src,
                  const std::vector<dealii::BlockVector<double>> &srcs,
                  const dealii::DoFHandler<dim> &dof_handler,
                  const aether::sn::QAngle<dim, qdim> &quadrature,
                  Mgxs &mgxs_rom);
  void enrich();
  void normalize();
  template <int zonesMD, int groupsMD, int zones1D, int zones2D>
  void setup(std::vector<InnerProducts2D1D<groupsMD, zonesMD>> coeffs_flux,
             const std::vector<double> &coeffs_src,
             const std::array<std::array<int, zones2D>, zones1D> &materials,
             const Mgxs &mgxs);
  void solve();
  void set_inner_prods();
  std::vector<Products> prods;
  std::vector<InnerProducts2D1D<groupsND, zonesND>> iprods_flux;
  std::vector<double> iprods_src;
 protected:
  void set_products(Products &prod);
  template <int zonesMD, int zones1D, int zones2D>
  void assert_zones_eq();
  void set_source(const std::vector<double> &coeffs_src);
  template <int zonesMD, int groupsMD, int zones1D, int zones2D>
  void set_residual(
      const std::vector<InnerProducts2D1D<groupsMD, zonesMD>> &coeffs_flux,
      const std::array<std::array<int, zones2D>, zones1D> &materials,
      const Mgxs &mgxs);
  template <int zonesMD, int groupsMD, int zones1D, int zones2D>
  void set_mgxs(const InnerProducts2D1D<groupsMD, zonesMD> &coeffs_xs,
                const std::array<std::array<int, zones2D>, zones1D> &materials,
                const Mgxs &mgxs);
  void solve_forward();
  void solve_adjoint();
  const aether::sn::FixedSource<dim, qdim> &fixed_src;
  const std::vector<dealii::BlockVector<double>> &srcs;
  Mgxs &mgxs_rom;
  dealii::BlockVector<double> flux;
  dealii::BlockVector<double> src;
  std::vector<int> dof_zones;
  dealii::Vector<double> scaling;
  const aether::sn::QAngle<qdim> &quadrature;
};

template <int dim, int qdim, int zones, int groups>
FixedSource2D1D<dim, qdim, zones, groups>::FixedSource2D1D(
    const aether::sn::FixedSource<dim, qdim> &fixed_src,
    const std::vector<dealii::BlockVector<double>> &srcs,
    const dealii::DoFHandler<dim> &dof_handler,
    const aether::sn::QAngle<dim, qdim> &quadrature,
    Mgxs &mgxs_rom) 
    : fixed_src(fixed_src), srcs(srcs), quadrature(quadrature), 
      dof_zones(dof_handler.n_dofs()), mgxs_rom(mgxs_rom) {
  std::vector<dealii::types::global_dof_index> dof_indices(
      dof_handler.get_fe().dofs_per_cell);
  for (const auto &cell : dof_handler.active_cell_iterators()) {
    if (!cell->is_locally_owned())
      continue;
    cell->get_dof_indices(dof_indices);
    for (dealii::types::global_dof_index i : dof_indices)
      dof_zones[i] = cell->material_id();
  }
}

template <int dim, int qdim, int zonesND, int groupsND>
void FixedSource2D1D<dim, qdim, zonesND, groupsND>::enrich() {
  flux = 1;
  Products prod = Products(groupsND, quadrature.size(), dof_zones.size());
  prod.psi = flux;
  set_products(prod);
  prods.push_back(prod);
}

template <int dim, int qdim, int zonesND, int groupsND>
void FixedSource2D1D<dim, qdim, zonesND, groupsND>::set_products(
    Products &prod) {
  for (int g = 0; g < groupsND; ++g) {
    fixed_src.d2m.vmult(prod.phi.block(g), prod.psi.block(g));
    // transport.stream(prod.streamed.block(g), prod.psi.block(g));
  }
}

template <int dim, int qdim, int zonesND, int groupsND>
template <int zonesMD, int groupsMD, int zones1D, int zones2D>
void FixedSource2D1D<dim, qdim, zonesND, groupsND>::setup(
    std::vector<InnerProducts2D1D<groupsMD, zonesMD>> coeffs_flux,
    const std::vector<double> &coeffs_src,
    const std::array<std::array<int, zones2D>, zones1D> &materials,
    const Mgxs &mgxs) {
  assert_zones_eq<zonesMD, zones1D, zones2D>();
  set_source(coeffs_src);
  auto coeffs_xs = coeffs_flux.pop_back();
  set_mgxs(coeffs_xs, mgxs);
  set_residual(coeffs_flux);
}

template <int dim, int qdim, int zonesND, int groupsND>
template <int zonesMD, int zones1D, int zones2D>
void FixedSource2D1D<dim, qdim, zonesND, groupsND>::assert_zones_eq() {
  if (dim == 1) {
    AssertDimension(zonesND, zones1D);
    AssertDimension(zonesMD, zones2D);
  } else {
    AssertDimension(zonesMD, zones1D);
    AssertDimension(zonesND, zones2D);
  }
}

template <int dim, int qdim, int zonesND, int groupsND>
void FixedSource2D1D<dim, qdim, zonesND, groupsND>::set_source(
    const std::vector<double> &coeffs_src) {
  AssertDimension(coeffs_src.size(), srcs.size());
  src = 0;
  for (int i = 0; i < srcs.size(); ++i)
    src.add(coeffs_src[i], srcs[i]);
}

template <int dim, int qdim, int zonesND, int groupsND>
template <int zonesMD, int groupsMD, int zones1D, int zones2D>
void FixedSource2D1D<dim, qdim, zonesND, groupsND>::set_residual(
    const std::vector<InnerProducts2D1D<groupsMD, zonesMD>> &coeffs_flux,
    const std::array<std::array<int, zones2D>, zones1D> &materials,
    const Mgxs &mgxs) {
  assert_zones_eq<zonesMD, zones1D, zones2D>();
  AssertDimension(coeffs_flux.size(), prods.size()-1);
  int groups = std::max(groupsND, groupsMD);
  bool mgND = groupsND > 1;
  bool mgMD = groupsMD > 1;
  if (mgND)
    AssertDimension(groups, groupsND);
  if (mgMD)
    AssertDimension(groups, groupsMD);
  for (int m = 0; m < prods.size(); ++m) {
    for (int g = 0; g < groups; ++g) {
      int gND = mgND ? g : 0;
      int gMD = mgMD ? g : 0;
      // subtract streamed flux
      for (int n = 0, nk = 0; n < quadrature.size(); ++n)
        for (int k = 0; k < dof_zones.size(); ++k, ++nk)
          src.block(gND)[nk] -= coeffs_flux.stream_co * 
                                prods[m].streamed.block(gND)[nk];
      for (int k = 0; k < scaling.size(); ++k) {
        scaling[k] = 0;
        for (int j = 0; j < zonesMD; ++j) {
          int matl = dim == 1 ? materials[j][dof_zones[k]]
                              : materials[dof_zones[k]][j];
          scaling[k] += mgxs.total[g][matl] * coeffs_flux.total[gMD][j];
        }
      }
      // subtract collided flux
      for (int n = 0, nk = 0; n < quadrature.size(); ++n) {
        double leakage_trans = dim == 1
            ? std::sqrt(1-std::pow(quadrature.angle(n)[0], 2))
            : quadrature.angle(n)[0];
        leakage_trans *= coeffs_flux.stream_trans;
        for (int k = 0; k < dof_zones.size(); ++k, ++nk)
          src.block(gND)[nk] -= (scaling[k] + leakage_trans) *
                                prods[m].psi.block(gND)[nk];
      }
      for (int gp = 0; gp < groups; ++gp) {
        int gpND = mgND ? gp : 0;
        int gpMD = mgMD ? gp : 0;
        for (int k = 0; k < scaling.size(); ++k) {
          scaling[k] = 0;
          for (int j = 0; j < zonesMD; ++j) {
            int matl = dim == 1 ? materials[j][dof_zones[k]]
                                : materials[dof_zones[k]][j];
            scaling[k] -= mgxs.scatter[g][gp][matl] *
                           coeffs_flux[gMD][gpMD][j];
          }
          // add (isotropic) scattered flux
          for (int n = 0, nk = 0; n < quadrature.size(); ++n)
            for (int k = 0; k < dof_zones.size(); ++k, ++nk) {
              src.block(gND)[nk] += scaling[k] *
                                    prods[m].phi.block(gpND)[k];
          }
        }
      }
    }
  }
}

template <int dim, int qdim, int zonesND, int groupsND>
template <int zonesMD, int groupsMD, int zones1D, int zones2D>
void FixedSource2D1D<dim, qdim, zonesND, groupsND>::set_mgxs(
    const InnerProducts2D1D<groupsMD, zonesMD> &coeffs_flux,
    const std::array<std::array<int, zones2D>, zones1D> &materials,
    const Mgxs &mgxs) {
  assert_zones_eq<zonesMD, zones1D, zones2D>();
  int groups = std::max(groupsND, groupsMD);
  bool mgND = groupsND > 1;
  bool mgMD = groupsMD > 1;
  if (mgND)
    AssertDimension(groups, groupsND);
  if (mgMD)
    AssertDimension(groups, groupsMD);
  for (int i = 0; i < zonesND; ++i) {
    for (int j = 0; j < zonesMD; ++j) {
      int matl = dim == 1 ? materials[j][i] : materials[i][j];
      for (int g = 0; g < groups; ++g) {
        int gND = mgND ? g : 0;
        int gMD = mgMD ? g : 0;
        mgxs_rom.total[gND][i] += mgxs.total[g][matl] *
                                  coeffs_flux.total[gMD][j];
        for (int gp = 0; gp < groups; ++gp) {
          int gpND = mgND ? gp : 0;
          int gpMD = mgMD ? gp : 0;
          mgxs_rom.scatter[gND][gpND][i] += mgxs.scatter[g][gp][matl] *
                                            coeffs_flux.scatter[gMD][gpMD][j];
        }
      }
    }
  }
}

}  // namespace aether::pgd::sn

#endif  // AETHER_PGD_SN_FIXED_SOURCE_2D1D_H_