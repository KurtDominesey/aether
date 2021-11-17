#include <deal.II/fe/fe_dgq.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/eigen.h>
#include <deal.II/lac/vector_memory.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function_lib.h>

#include "base/mgxs.h"
#include "sn/quadrature_lib.h"
#include "sn/fission_problem.h"
#include "sn/fission_source.h"
#include "sn/fixed_source_gs.h"

class Problem {
 public:
  Problem(const int &num_els, const int &num_ords, const double &length);
  int run_fixed_source();
  int run_criticality();
  void print_currents(const dealii::BlockVector<double> &flux);
  void plot(const dealii::BlockVector<double> &flux, const std::string &suffix)
      const;

 protected:
  static const int dim = 1;
  std::unique_ptr<aether::Mgxs> mgxs;
  dealii::Triangulation<dim> mesh;
  dealii::DoFHandler<dim> dof_handler;
  std::unique_ptr<aether::sn::QAngle<dim>> quadrature;
  using BoundaryConditions = 
      std::vector<std::vector<dealii::BlockVector<double> > >;
  BoundaryConditions boundary_conditions;
  std::unique_ptr<aether::sn::FissionProblem<dim>> problem;
};

template <int dim, typename number>
class Source : public dealii::Function<dim, number> {
 public:
  Source(double intensity, double xs, double mu)
    : intensity(intensity), xs(xs), mu(mu) {}
  virtual number value(const dealii::Point<dim> &p, const unsigned int = 0) 
      const override {
    return intensity * std::exp(-xs*(p[0]/mu));
  }

 protected:
  double intensity, xs, mu;
};