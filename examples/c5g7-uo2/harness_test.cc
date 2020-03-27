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
};