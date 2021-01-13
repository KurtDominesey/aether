#include "pgd/sn/fission_source_p.h"

namespace aether::pgd::sn {

template <int dim, int qdim>
FissionSourceP<dim, qdim>::FissionSourceP(
    aether::sn::FixedSource<dim, qdim> &fixed_source,
    aether::sn::Fission<dim, qdim> &fission,
    Mgxs &mgxs_pseudo, const Mgxs &mgxs)
    : FixedSourceP<dim, qdim>(fixed_source, mgxs_pseudo, mgxs, zero_sources), 
      fission(fission) {}

template <int dim, int qdim>
double FissionSourceP<dim, qdim>::step_eigenvalue(InnerProducts &coefficients) {
  this->set_cross_sections(coefficients);
  for (int j = 0; j < coefficients.fission.size(); ++j) {
    this->mgxs_pseudo.nu_fission[0][j] = 
        coefficients.fission[j] / coefficients.streaming;
  }
  const int num_groups = 1;
  auto &transport = this->fixed_source.within_groups[0].transport.transport;
  const int num_ords = transport.quadrature.size();
  const int num_dofs = transport.dof_handler.n_dofs();
  aether::PETScWrappers::BlockBlockWrapper fixed_source_petsc(
      num_groups, num_ords, MPI_COMM_WORLD, num_dofs, num_dofs, 
      this->fixed_source);
  aether::PETScWrappers::BlockBlockWrapper fission_petsc(
      num_groups, num_ords, MPI_COMM_WORLD, num_dofs, num_dofs, this->fission);
  const int size = num_dofs * num_ords * num_groups;
  std::vector<dealii::PETScWrappers::MPI::Vector> eigenvectors;
  eigenvectors.emplace_back(MPI_COMM_WORLD, size, size);
  eigenvectors[0].compress(dealii::VectorOperation::add);
  for (int i = 0; i < size; ++i) {
    eigenvectors[0][i] += this->caches.back().mode[i];
  }
  eigenvectors[0].compress(dealii::VectorOperation::add);
  // eigenvectors[0] /= eigenvectors[0].l2_norm();
  std::vector<double> eigenvalues = {1.0};
  dealii::IterationNumberControl control(50, 1e-6);
  dealii::SLEPcWrappers::SolverGeneralizedDavidson eigensolver(control);
  eigensolver.set_initial_space(eigenvectors);
  try {
    eigensolver.solve(fission_petsc, fixed_source_petsc, eigenvalues, 
                      eigenvectors);
  } catch (dealii::SolverControl::NoConvergence &failure) {
    failure.print_info(std::cout);
  }
  for (int i = 0; i < eigenvectors[0].size(); ++i)
    this->caches.back().mode[i] = eigenvectors[0][i];
  return eigenvalues[0];
}

template class FissionSourceP<1>;
template class FissionSourceP<2>;
template class FissionSourceP<3>;

}  // namespace aether::pgd::sn