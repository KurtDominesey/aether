#ifndef AETHER_PGD_SN_FIXED_SOURCE_2D1D_H_
#define AETHER_PGD_SN_FIXED_SOURCE_2D1D_H_

#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_richardson.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/precondition.h>

#include "base/mgxs.h"
#include "sn/fixed_source.h"
#include "sn/quadrature.h"
#include "pgd/sn/transport.h"
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

template <int dim, int qdim = dim == 1 ? 1 : 2>
class FixedSource2D1D {
 public:
  FixedSource2D1D(const aether::sn::FixedSource<dim, qdim> &fixed_src,
                  const std::vector<dealii::BlockVector<double>> &srcs,
                  const Transport<dim, qdim> &transport,
                  Mgxs &mgxs_rom);
  void enrich();
  void normalize();
  void setup(std::vector<InnerProducts2D1D> coeffs_flux,
             const std::vector<double> &coeffs_src,
             const std::vector<std::vector<int>> &materials,
             const Mgxs &mgxs);
  double solve();
  void set_inner_prods();
  std::vector<Products> prods;
  std::vector<InnerProducts2D1D> iprods_flux;
  std::vector<double> iprods_src;
 protected:
  void set_products(Products &prod);
  void set_inner_prod_flux(const dealii::BlockVector<double> &test,
                           const Products &prod,
                           InnerProducts2D1D &iprod);
  double inner_prod_src(const dealii::BlockVector<double> &test,
                        const dealii::BlockVector<double> &src);
  void set_source(const std::vector<double> &coeffs_src,
                  const std::vector<double> &denom);
  void set_residual(
      const std::vector<InnerProducts2D1D> &coeffs_flux,
      const std::vector<std::vector<int>> &materials,
      const Mgxs &mgxs);
  void set_mgxs(const InnerProducts2D1D &coeffs_xs,
                const std::vector<std::vector<int>> &materials,
                const Mgxs &mgxs);
  void check_mgxs();
  void solve_forward();
  void solve_adjoint();
  const aether::sn::FixedSource<dim, qdim> &fixed_src;
  const std::vector<dealii::BlockVector<double>> &srcs;
  Mgxs &mgxs_rom;
  dealii::BlockVector<double> src;
  dealii::BlockVector<double> uncollided;
  std::vector<int> dof_zones;
  dealii::Vector<double> scaling;
  const Transport<dim, qdim> &transport;
};

}  // namespace aether::pgd::sn

#endif  // AETHER_PGD_SN_FIXED_SOURCE_2D1D_H_