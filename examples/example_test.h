#ifndef AETHER_EXAMPLES_EXAMPLE_TEST_H_
#define AETHER_EXAMPLES_EXAMPLE_TEST_H_

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

template <int dim, int qdim = dim == 1 ? 1 : 2>
class ExampleTest : public ::testing::Test {
 protected:
  dealii::Triangulation<dim> mesh;
  dealii::DoFHandler<dim> dof_handler;
  QAngle<qdim> quadrature;
  std::shared_ptr<Mgxs> mgxs;

  std::string GetTestName() {
    const ::testing::TestInfo* const test_info =
        ::testing::UnitTest::GetInstance()->current_test_info();
    std::string name = test_info->name();
    auto this_with_param = 
      dynamic_cast<::testing::WithParamInterface<std::string>*>(this);
    if (this_with_param != nullptr)
      name = std::regex_replace(name, std::regex("/[0-9]+"), 
                                "/" + this_with_param->GetParam());
    std::string case_name = test_info->test_case_name();
    case_name += name;
    case_name = std::regex_replace(case_name, std::regex("/"), "_");
    return case_name;
  }

  void WriteConvergenceTable(dealii::ConvergenceTable &table,
                             const std::string suffix="") {
    std::string filename = GetTestName();
    filename += suffix;
    filename += ".txt";
    std::ofstream out(filename, std::ofstream::out | std::ofstream::trunc);
    table.write_text(out);
    out.close();
  }
};

#endif  // AETHER_EXAMPLES_EXAMPLE_TEST_H_