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
  FixedSourceP<dim, qdim>::set_cross_sections(coefficients);
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
  this->eigenvalue = eigenvalues[0];
  return eigenvalues[0];
}

template <int dim, int qdim>
void FissionSourceP<dim, qdim>::set_cross_sections(
    const InnerProducts &coefficients) {
  FixedSourceP<dim, qdim>::set_cross_sections(coefficients);
  const int num_groups = this->mgxs.total.size();
  const int num_materials = this->mgxs.total[0].size();
  for (int g = 0; g < num_groups; ++g) {
    for (int gp = 0; gp < num_groups; ++gp) {
      for (int j = 0; j < num_materials; ++j) {
        this->mgxs_pseudo.scatter[g][gp][j] += this->mgxs.chi[g][j]
                                               * this->mgxs.nu_fission[gp][j]
                                               * coefficients.fission[j]
                                               / this->eigenvalue
                                               / coefficients.streaming;
      }
    }
  }
}

template <int dim, int qdim>
void FissionSourceP<dim, qdim>::subtract_modes_from_source(
    dealii::BlockVector<double> &source,
    std::vector<InnerProducts> coefficients) {
  FixedSourceP<dim, qdim>::subtract_modes_from_source(source, coefficients);
  const int num_groups = this->fixed_source.within_groups.size();
  const auto &transport = 
      this->fixed_source.within_groups[0].transport.transport;
  std::vector<dealii::types::global_dof_index> dof_indices(
      transport.dof_handler.get_fe().dofs_per_cell);
  for (int m = 0; m < this->caches.size() - 1; ++m) {
    dealii::Vector<double> produced(transport.dof_handler.n_dofs());
    for (int gp = 0; gp < num_groups; ++gp) {
      for (auto cell = transport.dof_handler.begin_active();
           cell != transport.dof_handler.end(); ++cell) {
        cell->get_dof_indices(dof_indices);
        int material = cell->material_id();
        for (int i = 0; i < dof_indices.size(); ++i) {
          produced[dof_indices[i]] -= 
              this->mgxs.nu_fission[gp][material] 
              * coefficients[m].fission[material]
              * this->caches[m].moments.block(gp)[dof_indices[i]];
        }
      }
    }
    for (int g = 0; g < num_groups; ++g) {
      dealii::Vector<double> emitted(produced);
      for (auto cell = transport.dof_handler.begin_active();
           cell != transport.dof_handler.end(); ++cell) {
        cell->get_dof_indices(dof_indices);
        int material = cell->material_id();
        for (int i = 0; i < dof_indices.size(); ++i) {
          emitted[dof_indices[i]] *= this->mgxs.chi[g][material] 
                                     / this->eigenvalue;
        }
      }
      this->fixed_source.m2d.vmult_add(source.block(g), emitted);
    }
  }
}

template class FissionSourceP<1>;
template class FissionSourceP<2>;
template class FissionSourceP<3>;

}  // namespace aether::pgd::sn