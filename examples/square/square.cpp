#include <deal.II/lac/vector.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/lac/solver_richardson.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/vector_tools.h>

#include "../../src/sn/conservation.hpp"

enum Term : char {
  STREAMING = 'L',
  COLLISION = 'T',
  SCATTERING = 'S'
};

int main () {
  // discretize the problem
  int degree = 1;
  int num_ords = 4;
  const int dim = 2;
  const int qdim = 2;
  const int num_octs = std::pow(dim, 2);
  dealii::FE_DGQ<dim> fe(degree);
  dealii::Triangulation<dim> mesh;
  dealii::GridGenerator::subdivided_hyper_cube(mesh, 20, -1, 1);
  dealii::QGauss<qdim> quadrature(num_ords);
  // assign material properties
  std::map<char, std::vector<double> > cross_sections;
  cross_sections[COLLISION] = {1};
  cross_sections[SCATTERING] = {1};
  // initialize equation
  Conservation<dim, qdim> conservation(fe, quadrature, mesh, cross_sections);
  // set up sources
  dealii::Functions::PillowFunction<dim> source_iso;
  dealii::Vector<double> source_iso_h(conservation.dof_handler.n_dofs());
  dealii::VectorTools::create_right_hand_side(conservation.dof_handler,
                                              conservation.quadrature_fe,
                                              source_iso, source_iso_h);
  dealii::BlockVector<double> source_ang(num_octs * num_ords,
                                         source_iso_h.size());
  for (int oct = 0; oct < num_octs; ++oct) {
    for (int ord = 0; ord < num_ords; ++ord) {
      source_ang.block(oct*num_ords+ord) = source_iso_h;
    }
  }
  // solve the one-group iterations
  int max_iters = 10;
  double tol = 1e-6;
  dealii::SolverControl solver_control(max_iters, tol);
  dealii::SolverRichardson<dealii::BlockVector<double> > solver(solver_control);
  dealii::BlockVector<double> flux_ang(source_ang.get_block_indices());
  dealii::BlockVector<double> source_lm(1, source_iso_h.size());
  source_lm.block(0) = source_iso_h;
  dealii::BlockVector<double> flux_lm(source_lm.get_block_indices());
  flux_lm = 0;
  solver.solve(conservation, flux_lm, source_lm,
               dealii::PreconditionIdentity());
  std::cout << "done" << std::endl;
  std::cout << source_lm.l2_norm() << std::endl;
  std::cout << flux_lm.l2_norm() << std::endl;
  std::cout << (flux_lm == source_lm) << std::endl;
}