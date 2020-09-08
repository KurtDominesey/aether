#ifndef AETHER_BASE_TESTS_TESTMATRIX_H_
#define AETHER_BASE_TESTS_TESTMATRIX_H_

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/numerics/matrix_tools.h>

namespace aether {

void create_laplace_matrix(dealii::SparseMatrix<double> &matrix,
                           dealii::SparsityPattern &sparsity_pattern,
                           const int size) {
  dealii::Triangulation<2> mesh;
  dealii::GridGenerator::subdivided_hyper_cube(mesh, size);
  dealii::FE_Q<2> fe(1);
  dealii::DoFHandler<2> dof_handler(mesh);
  dof_handler.distribute_dofs(fe);
  dealii::QGauss<2> quadrature(fe.degree + 1);
  dealii::DynamicSparsityPattern dsp(dof_handler.n_dofs());
  dealii::DoFTools::make_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);
  matrix.reinit(sparsity_pattern);
  dealii::MatrixCreator::create_laplace_matrix(dof_handler, quadrature, matrix);
}

}  // namespace aether

#endif  // AETHER_BASE_TESTS_TESTMATRIX_H_