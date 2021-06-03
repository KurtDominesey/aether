#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_richardson.h>
#include <deal.II/lac/eigen.h>
#include <deal.II/lac/vector_memory.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/slepc_solver.h>
#include <deal.II/lac/petsc_precondition.h>

#include "base/mgxs.h"
#include "base/petsc_block_block_wrapper.h"
#include "mesh/mesh.h"
#include "sn/quadrature_lib.h"
#include "sn/fission_problem.h"
#include "sn/fission_source.h"
#include "sn/fixed_source_gs.h"
#include "gtest/gtest.h"

namespace aether::sn {

namespace {

template <int dim>
class CriticalityTest : public testing::Test {
 protected:
  std::unique_ptr<Mgxs> mgxs;
  dealii::Triangulation<dim> mesh;
  dealii::DoFHandler<dim> dof_handler;
  std::unique_ptr<QAngle<dim>> quadrature;
  using BoundaryConditions = 
      std::vector<std::vector<dealii::BlockVector<double> > >;
  BoundaryConditions boundary_conditions;
  std::unique_ptr<FissionProblem<dim>> problem;
};

class SoodTest : public CriticalityTest<1> {
 protected:
  static const int dim = 1;
  void SetUp() override {
    // Pu-a cross-sections
    double length = 0;  // cm, critical slab half-length
    const int num_groups = 2;
    mgxs = std::make_unique<Mgxs>(num_groups, 1, 1);
    if (num_groups == 1) {
      length = 1.853722;
      mgxs->nu_fission[0][0] = 3.24 * 0.081600;
      mgxs->scatter[0][0][0] = 0.225216;
      mgxs->total[0][0] = 0.32640;
      mgxs->chi[0][0] = 1;
    } else if (num_groups == 2) {
      length = 1.795602;
      mgxs->nu_fission[0][0] = 3.10 * 0.0936;
      mgxs->nu_fission[1][0] = 2.93 * 0.08544;
      mgxs->scatter[0][0][0] = 0.0792;
      mgxs->scatter[0][1][0] = 0.0;  // no upscattering
      mgxs->scatter[1][0][0] = 0.0432;
      mgxs->scatter[1][1][0] = 0.23616;
      mgxs->total[0][0] = 0.2208;
      mgxs->total[1][0] = 0.3360;
      mgxs->chi[0][0] = 0.575;
      mgxs->chi[1][0] = 0.425;
    } else {
      AssertThrow(false, dealii::ExcInvalidState());
    }
    const int dim = 1;
    dealii::FE_DGQ<dim> fe(1);
    dealii::GridGenerator::subdivided_hyper_cube(mesh, 50, -length, length);
    dof_handler.initialize(mesh, fe);
    quadrature = std::make_unique<QPglc<dim>>(128);
    // vacuum boundary conditions (for 1 group and 2 boundaries)
    bool reflecting = false;
    if (reflecting)
      set_all_boundaries_reflecting(mesh);
    int num_boundaries = reflecting ? 0 : (dim == 1 ? 2 : 1);
    boundary_conditions.resize(num_groups, 
        std::vector<dealii::BlockVector<double>>(num_boundaries,
          dealii::BlockVector<double>(quadrature->size(), fe.dofs_per_cell)));
    problem = std::make_unique<FissionProblem<dim>>(
        dof_handler, *quadrature, *mgxs, boundary_conditions);
  }
};

class TakedaOneTest : public CriticalityTest<3> {
 protected:
  static const int dim = 3;
  void SetUp() override {
    // set up materials
    const int num_groups = 2;
    const int num_materials = 4;  // core, reflector, control rod, void
    mgxs = std::make_unique<Mgxs>(num_groups, num_materials, 1);
    mgxs->total[0] = {2.23775e-1, 2.50367e-1, 8.52325e-2, 1.28407e-2};
    mgxs->total[1] = {1.03864,    1.64482,    2.17460e-1, 1.20676e-2};
    mgxs->scatter[0][0] = {1.92423e-1, 1.93446e-1, 6.77241e-2, 1.27700e-2};
    mgxs->scatter[0][1] = {0.0, 0.0, 0.0, 0.0};
    mgxs->scatter[1][0] = {2.28253e-2, 5.65042e-2, 6.45461e-5, 2.40997e-5};
    mgxs->scatter[1][1] = {8.80439e-1, 1.62452,    3.52358e-2, 1.07387e-2};
    mgxs->nu_fission[0] = {9.09319e-3, 0.0, 0.0, 0.0};
    mgxs->nu_fission[1] = {2.90183e-1, 0.0, 0.0, 0.0};
    mgxs->chi[0] = {1.0, 0.0, 0.0, 0.0};
    mgxs->chi[1] = {0.0, 0.0, 0.0, 0.0};
    // set up mesh
    // In deal.II 9.1.1, the simpler subdivided_hyper_cube does not support
    // colorization. This in rectified in 9.2.0, and so should be updated later.
    dealii::GridGenerator::subdivided_hyper_rectangle(mesh, {25, 25, 25}, 
        dealii::Point<dim>(0, 0, 0), dealii::Point<dim>(25, 25, 25), true);
    bool rodded = true;
    for (auto &cell : mesh.active_cell_iterators()) {
      const dealii::Point<dim> &center = cell->center();
      bool outside_core = center[0] > 15 || center[1] > 15 || center[2] > 15;
      bool in_rod = (center[0] > 15 && center[0] < 20) && center[1] < 5;
      if (in_rod)
        cell->set_material_id(rodded ? 2 : 3);
      else if (outside_core)
        cell->set_material_id(1);
      else
        cell->set_material_id(0);
      for (int f = 0; f < dealii::GeometryInfo<dim>::faces_per_cell; ++f) {
        // replace interior (0, 2, 4) with reflecting
        // replace exterior (1, 3, 5) with vacuum (0)
        dealii::types::boundary_id b = cell->face(f)->boundary_id();
        if (b == 0 || b == 2 || b == 4)
          cell->face(f)->set_boundary_id(types::reflecting_boundary_id);
        else if (b == 1 || b == 3 || b == 5)
          cell->face(f)->set_boundary_id(0);
        else
          AssertThrow(b == dealii::numbers::internal_face_boundary_id, 
                      dealii::ExcInvalidState());
      }
    }
    // write out the mesh
    dealii::GridOut grid_out;
    std::ofstream vtk_out("takeda1.vtk");
    grid_out.write_vtk(mesh, vtk_out);
    // set up finite elements and quadrature
    dealii::FE_DGP<dim> fe(1);
    // mesh.refine_global();
    dof_handler.initialize(mesh, fe);
    quadrature = std::make_unique<QPglc<dim>>(2, 2);
    double sum = 0;
    for (int n = 0; n < quadrature->size(); ++n)
      sum += quadrature->weight(n);
    AssertThrow(std::abs(sum - 1.0) < 1e-6, dealii::ExcInvalidState());
    // set up (vacuum) boundary conditions
    boundary_conditions.resize(num_groups,
        std::vector<dealii::BlockVector<double>>(1, 
          dealii::BlockVector<double>(quadrature->size(), fe.dofs_per_cell)));
    // set up the fission problem
    problem = std::make_unique<FissionProblem<dim>>(
        dof_handler, *quadrature, *mgxs, boundary_conditions);
  }
};

class TakedaTwoTest : public CriticalityTest<3> {
 protected:
  static const int dim = 3;
  void SetUp() override {
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
    mgxs = std::make_unique<Mgxs>(num_groups, num_materials, 1);
    for (int j = 0; j < num_materials; ++j) {
      for (int g = 0; g < num_groups; ++g) {
        mgxs->total[g][j] = total[j][g];
        mgxs->nu_fission[g][j] = nu_fission[j][g];
        mgxs->chi[g][j] = j <= 2 ? chi[g] : 0;
        for (int gp = 0; gp < num_groups; ++gp)
          mgxs->scatter[g][gp][j] = scatter[j][g][gp];
      }
    }
    // set up mesh
    dealii::GridGenerator::subdivided_hyper_rectangle(mesh, {14, 14, 30}, 
        dealii::Point<dim>(0, 0, 0), dealii::Point<dim>(70, 70, 150), true);
    for (auto &cell : mesh.active_cell_iterators()) {
      const dealii::Point<dim> &center = cell->center();
      // precedence: in_rod, in_radial_blanket, in_axial_blanket
      bool in_rod = (center[0] > 35 && center[0] < 45) && center[1] < 5;
      bool in_radial_blanket = 
          center[0] > 55 || center[1] > 55 ||  // outside core
          (center[0] > 50 && center[0] < 55 && center[1] > 15) ||  // last row
          (center[1] > 50 && center[1] < 55 && center[0] > 15) ||  // last col
          (center[0] > 45 && center[0] < 50 && center[1] > 30) ||  // second row
          (center[1] > 45 && center[1] < 50 && center[0] > 30) ||  // second col
          (center[0] > 40 && center[0] < 45 && center[1] > 40) ||  // third row
          (center[1] > 40 && center[1] < 45 && center[0] > 40);    // third col
      bool in_axial_blanket = center[2] < 20 || center[2] > 130;
      if (in_rod)
        cell->set_material_id(center[2] > 75 ? 3 : 4);
      else if (in_radial_blanket)
        cell->set_material_id(1);
      else if (in_axial_blanket)
        cell->set_material_id(2);
      else
        cell->set_material_id(0);  // core
      for (int f = 0; f < dealii::GeometryInfo<dim>::faces_per_cell; ++f) {
        // replace interior x-y (0, 2) with reflecting
        // replace exterior x-y (1, 3) and z (4, 5) with vacuum (0)
        dealii::types::boundary_id b = cell->face(f)->boundary_id();
        if (b == 0 || b == 2)
          cell->face(f)->set_boundary_id(types::reflecting_boundary_id);
        else if (b == 1 || b == 3 || b == 4 || b == 5)
          cell->face(f)->set_boundary_id(0);
        else
          AssertThrow(b == dealii::numbers::internal_face_boundary_id, 
                      dealii::ExcInvalidState());
      }
    }
    // write out the mesh
    dealii::GridOut grid_out;
    std::ofstream vtk_out("takeda2.vtk");
    grid_out.write_vtk(mesh, vtk_out);
    // set up finite elements and quadrature
    dealii::FE_DGP<dim> fe(1);
    dof_handler.initialize(mesh, fe);
    quadrature = std::make_unique<QPglc<dim>>(2, 2);
    // set up (vacuum) boundary conditions
    boundary_conditions.resize(num_groups,
        std::vector<dealii::BlockVector<double>>(1, 
          dealii::BlockVector<double>(quadrature->size(), fe.dofs_per_cell)));
    // set up the fission problem
    problem = std::make_unique<FissionProblem<dim>>(
        dof_handler, *quadrature, *mgxs, boundary_conditions);
  }
};

TEST_F(TakedaOneTest, PuAOneGroupIsotropicSlab) {
  const int num_groups = this->mgxs->total.size();
  // dealii::ReductionControl control_wg(100, 1e-10, 1e-2);
  // dealii::SolverGMRES<dealii::Vector<double>> solver_wg(control_wg);
  // FixedSourceGS fixed_source_gs(this->problem->fixed_source, solver_wg);
  dealii::ReductionControl control_fs(100, 1e-6, 1e-2);
  dealii::SolverGMRES<dealii::BlockVector<double>> solver_fs(control_fs);
  FissionSource fission_source(
      this->problem->fixed_source, this->problem->fission, solver_fs, 
      dealii::PreconditionIdentity());
  dealii::SolverControl control(100, 1e-5);
  const int size = this->dof_handler.n_dofs() * this->quadrature->size();
  dealii::GrowingVectorMemory<dealii::BlockVector<double>> memory;
  dealii::EigenPower<dealii::BlockVector<double>> eigensolver(control, memory);
  double k = 1.0;  // bad initial guess
  dealii::BlockVector<double> flux(num_groups, size);
  flux = 1;
  flux /= flux.l2_norm();
  eigensolver.solve(k, fission_source, flux);
  std::cout << k << ", " << flux.l2_norm() << "\n";
  EXPECT_NEAR(k, 1, 1e-5);  // within one pcm of criticality
  // plot flux
  dealii::DataOut<this->dim> data_out;
  data_out.attach_dof_handler(this->dof_handler);
  std::vector<dealii::Vector<double>> fluxes(num_groups,
      dealii::Vector<double>(this->dof_handler.n_dofs()));
  // quick and sloppy way to compute scalar fluxes
  for (int g = 0; g < num_groups; ++g) {
    data_out.add_data_vector(fluxes[g], "g"+std::to_string(g));
    int gg = g * this->quadrature->size() * this->dof_handler.n_dofs();
    for (int n = 0; n < this->quadrature->size(); ++n) {
      int nn = n * this->dof_handler.n_dofs();
      for (int i = 0; i < this->dof_handler.n_dofs(); ++i) {
        fluxes[g][i] += this->quadrature->weight(n) * flux[gg+nn+i];
      }
    }
    if (fluxes[g][0] < 0)
      fluxes[g] *= -1;
  }
  data_out.build_patches();
  std::ofstream output("takeda1-dealii.vtu");
  data_out.write_vtu(output);
}

template <typename Solver>
class SoodSLEPcTest : public SoodTest {};

using Solvers = ::testing::Types<
    dealii::SLEPcWrappers::SolverPower,
    dealii::SLEPcWrappers::SolverArnoldi,
    dealii::SLEPcWrappers::SolverKrylovSchur,
    dealii::SLEPcWrappers::SolverGeneralizedDavidson,
    dealii::SLEPcWrappers::SolverJacobiDavidson
>;
TYPED_TEST_CASE(SoodSLEPcTest, Solvers);

TYPED_TEST(SoodSLEPcTest, PuAOneGroupIsotropicSlab) {
  const int num_groups = this->mgxs->total.size();
  dealii::ReductionControl control_wg(100, 1e-5, 1e-3);
  dealii::SolverGMRES<dealii::BlockVector<double>> solver_wg(control_wg);
  PETScWrappers::FissionSource fission_source(
      this->problem->fixed_source, this->problem->fission, solver_wg, 
      dealii::PreconditionIdentity());
  dealii::SolverControl control(100, 1e-5);
  const int size = 
      this->dof_handler.n_dofs() * this->quadrature->size() * num_groups;
  std::vector<dealii::PETScWrappers::MPI::Vector> eigenvectors;
  eigenvectors.emplace_back(MPI_COMM_WORLD, size, size);
  std::vector<double> eigenvalues = {0.5};
  TypeParam eigensolver(control);
  eigensolver.solve(fission_source, eigenvalues, eigenvectors, 1);
  EXPECT_NEAR(eigenvalues[0], 1, 1e-5);
}

TYPED_TEST(SoodSLEPcTest, PuAOneGroupIsotropicSlabGeneralized) {
  const int num_groups = this->mgxs->total.size();
  ::aether::PETScWrappers::BlockBlockWrapper fixed_source(
      num_groups, this->quadrature->size(), MPI_COMM_WORLD, 
      this->dof_handler.n_dofs(), this->dof_handler.n_dofs(),
      this->problem->fixed_source);
  ::aether::PETScWrappers::BlockBlockWrapper fission(
      num_groups, this->quadrature->size(), MPI_COMM_WORLD, 
      this->dof_handler.n_dofs(), this->dof_handler.n_dofs(),
      this->problem->fission);
  dealii::SolverControl control(100, 1e-5);
  const int size = 
      this->dof_handler.n_dofs() * this->quadrature->size() * num_groups;
  std::vector<dealii::PETScWrappers::MPI::Vector> eigenvectors;
  eigenvectors.emplace_back(MPI_COMM_WORLD, size, size);
  eigenvectors[0] = 1;
  eigenvectors[0] /= eigenvectors[0].l2_norm();
  std::vector<double> eigenvalues = {0.5};
  TypeParam eigensolver(control);
  eigensolver.set_initial_space(eigenvectors);
  bool is_davidson = 
      dynamic_cast<dealii::SLEPcWrappers::SolverGeneralizedDavidson*>
      (&eigensolver) != nullptr ||
      dynamic_cast<dealii::SLEPcWrappers::SolverJacobiDavidson*>
      (&eigensolver) != nullptr;
  if (is_davidson) {
    eigensolver.solve(fission, fixed_source, eigenvalues, eigenvectors);
  } else {
    dealii::SLEPcWrappers::TransformationShiftInvert shift_invert(
        MPI_COMM_WORLD,
        dealii::SLEPcWrappers::TransformationShiftInvert::AdditionalData(0.9));
    shift_invert.set_matrix_mode(ST_MATMODE_SHELL);
    dealii::ReductionControl control_inv(100, 1e-5, 1e-3);
    dealii::PETScWrappers::SolverGMRES solver_inv(control_inv, MPI_COMM_WORLD);
    dealii::PETScWrappers::PreconditionNone preconditioner(fixed_source);
    solver_inv.initialize(preconditioner);
    shift_invert.set_solver(solver_inv);
    eigensolver.set_transformation(shift_invert);
    eigensolver.solve(fission, fixed_source, eigenvalues, eigenvectors);
  }
  // multiplication factor (k) is reciprocal of eigenvalue (lambda)
  // for (int i = 0; i < eigenvalues.size(); ++i)
  //   eigenvalues[i] = 1.0 / eigenvalues[i];
  EXPECT_NEAR(eigenvalues[0], 1, 1e-5);
  // plot flux
  dealii::DataOut<this->dim> data_out;
  data_out.attach_dof_handler(this->dof_handler);
  std::vector<dealii::Vector<double>> fluxes(num_groups,
      dealii::Vector<double>(this->dof_handler.n_dofs()));
  // quick and sloppy way to compute scalar fluxes
  for (int g = 0; g < num_groups; ++g) {
    data_out.add_data_vector(fluxes[g], "g"+std::to_string(g));
    int gg = g * this->quadrature->size() * this->dof_handler.n_dofs();
    for (int n = 0; n < this->quadrature->size(); ++n) {
      int nn = n * this->dof_handler.n_dofs();
      for (int i = 0; i < this->dof_handler.n_dofs(); ++i) {
        fluxes[g][i] += this->quadrature->weight(n) * eigenvectors[0][gg+nn+i];
      }
    }
    if (fluxes[g][0] < 0)
      fluxes[g] *= -1;
  }
  data_out.build_patches();
  std::ofstream output("takeda1.vtu");
  data_out.write_vtu(output);
}

}  // namespace

}  // namespace aether::sn