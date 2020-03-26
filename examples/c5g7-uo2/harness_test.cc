#include <regex>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_dof_data.h>

#include "mesh/mesh.h"
#include "sn/transport.h"
#include "sn/transport_block.h"
#include "sn/scattering.h"
#include "sn/scattering_block.h"
#include "sn/within_group.h"
#include "sn/discrete_to_moment.h"
#include "sn/moment_to_discrete.h"
#include "sn/quadrature.h"
#include "sn/quadrature_lib.h"
#include "types/types.h"
#include "base/mgxs.h"
#include "functions/function_lib.h"
#include "sn/fixed_source.h"
#include "sn/fixed_source_problem.h"
#include "sn/fixed_source_gs.h"
#include "pgd/sn/energy_mg_full.h"
#include "pgd/sn/fixed_source_p.h"
#include "pgd/sn/nonlinear_gs.h"

#include "gtest/gtest.h"

using namespace aether;
using namespace aether::sn;

class C5G7Test : public ::testing::Test {
 protected:
  static const int dim = 2;
  static const int qdim = 2;
  const double pitch = 0.63;
  dealii::Triangulation<dim> mesh;
  dealii::DoFHandler<dim> dof_handler;
  QAngle<qdim> quadrature;
  std::shared_ptr<Mgxs> mgxs;

  void SetUp() override {
    const std::vector<std::string> materials =  {"water", "uo2"};
    mgxs = std::make_shared<Mgxs>(7, materials.size(), 1);
    read_mgxs(*mgxs, "c5g7.h5", "294K", materials);
    quadrature = QPglc<qdim>(1, 2);
    mesh_quarter_pincell(mesh, {0.54}, pitch, {0, 1});
    set_all_boundaries_reflecting(mesh);
    mesh.refine_global(0);
  }

  void WriteConvergenceTable(dealii::ConvergenceTable &table) {
    const ::testing::TestInfo* const test_info =
        ::testing::UnitTest::GetInstance()->current_test_info();
    std::string filename = test_info->test_case_name();
    filename += test_info->name();
    filename += ".txt";
    filename = std::regex_replace(filename, std::regex("/"), "_");
    std::ofstream out(filename, std::ofstream::trunc);
    table.write_text(out);
    out.close();
  }

  template <int dim, int qdim, class Solution>
  void CreateSource(
      dealii::BlockVector<double> &source,
      const std::vector<Solution> &solutions_spaceangle,
      const std::vector<dealii::Vector<double>> &solutions_energy,
      const dealii::DoFHandler<dim> &dof_handler,
      const QAngle<qdim> &quadrature,
      const Mgxs &mgxs) {
    std::vector<dealii::BlockVector<double>> sources_spaceangle;
    std::vector<dealii::Vector<double>> sources_energy;
    CreateSources(sources_spaceangle, sources_energy,
                  solutions_spaceangle, solutions_energy,
                  dof_handler, quadrature, mgxs);
    const int num_sources = sources_spaceangle.size();
    AssertDimension(num_sources, sources_energy.size());
    const int num_groups = mgxs.total.size();
    for (int s = 0; s < num_sources; ++s)
      for (int g = 0; g < num_groups; ++g)
        source.block(g).add(sources_energy[s][g], sources_spaceangle[s].block(0));
  }

  template <int dim, int qdim, class Solution>
  void CreateSources(
        std::vector<dealii::BlockVector<double>> &sources_spaceangle,
        std::vector<dealii::Vector<double>> &sources_energy,
        const std::vector<Solution> &solutions_spaceangle,
        const std::vector<dealii::Vector<double>> &solutions_energy,
        const dealii::DoFHandler<dim> &dof_handler,
        const QAngle<qdim> &quadrature,
        const Mgxs &mgxs) {
    const int num_materials = mgxs.total[0].size();
    AssertDimension(solutions_spaceangle.size(), solutions_energy.size());
    AssertDimension(sources_spaceangle.size(), 0);
    AssertDimension(sources_energy.size(), 0);
    const int num_groups = mgxs.total.size();
    const int num_solutions = solutions_spaceangle.size();
    const int num_sources_per_solution = 1 + 2 * num_materials;
    const int num_sources = num_solutions * num_sources_per_solution;
    sources_spaceangle.resize(num_sources, 
        dealii::BlockVector<double>(1, quadrature.size()*dof_handler.n_dofs()));
    sources_energy.resize(num_sources, dealii::Vector<double>(num_groups));
    for (int i = 0; i < num_solutions; ++i) {
      const int s = i * num_sources_per_solution;
      dealii::BlockVector<double> source_spaceangle(
          quadrature.size(), dof_handler.n_dofs());
      auto grad = std::bind(&Solution::gradient, solutions_spaceangle[i], 
                            std::placeholders::_1, 0);
      for (int n = 0; n < quadrature.size(); ++n) {
        Streamed<dim> streamed(quadrature.ordinate(n), grad);
        dealii::VectorTools::interpolate(
            dof_handler, streamed, source_spaceangle.block(n));
      }
      sources_spaceangle[s].block(0) = source_spaceangle;
      sources_energy[s] = solutions_energy[i];
      // collision and scattering
      dealii::Vector<double> source_spaceangle_iso(dof_handler.n_dofs());
      dealii::VectorTools::interpolate(
          dof_handler, solutions_spaceangle[i], source_spaceangle_iso);
      for (int j = 0; j < num_materials; ++j) {
        for (int g = 0; g < num_groups; ++g) {
          sources_energy[s+1+j][g] = mgxs.total[g][j] 
                                      * solutions_energy[i][g];
          for (int gp = 0; gp < num_groups; ++gp)
            sources_energy[s+1+num_materials+j][g] += mgxs.scatter[g][gp][j]
                                                      * solutions_energy[i][gp];
        }
        dealii::Vector<double> source_spaceangle_iso_j(source_spaceangle_iso);
        std::vector<dealii::types::global_dof_index> dof_indices(
            dof_handler.get_fe().dofs_per_cell);
        for (auto cell = dof_handler.begin_active();
              cell != dof_handler.end(); ++cell) {
          if (cell->material_id() != j) {
            cell->get_dof_indices(dof_indices);
            for (auto index : dof_indices) {
              source_spaceangle_iso_j[index] = 0;
            }
          }
        }
        source_spaceangle = 0;
        for (int n = 0; n < quadrature.size(); ++n)
          source_spaceangle.block(n) = source_spaceangle_iso_j;
        sources_spaceangle[s+1+j].block(0) = source_spaceangle;
        sources_spaceangle[s+1+num_materials+j].block(0) = source_spaceangle;
        sources_spaceangle[s+1+num_materials+j].block(0) *= -1;
      }
    }
  }
};