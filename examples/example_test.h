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
  std::unique_ptr<Mgxs> mgxs;

  std::string GetTestName() const {
    const ::testing::TestInfo* const test_info =
        ::testing::UnitTest::GetInstance()->current_test_info();
    std::string param = "";
    std::string name = test_info->name();
    auto this_with_param = 
      dynamic_cast<const ::testing::WithParamInterface<std::string>*>(this);
    if (this_with_param != nullptr)
      param = this_with_param->GetParam();
    else {  // TODO: generalize this
      auto this_with_params =
          dynamic_cast<const ::testing::WithParamInterface<
            std::tuple<std::string, std::string>>*>(this);
      if (this_with_params != nullptr)
        param = std::get<0>(this_with_params->GetParam())
                + "/" + std::get<1>(this_with_params->GetParam());
    }
    if (!param.empty())
      name = std::regex_replace(name, std::regex("/[0-9]+"), "/" + param);
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

  void PlotFlux(const dealii::BlockVector<double> &flux,
                const DiscreteToMoment<dim, qdim> &d2m,
                const std::vector<double> &group_structure,
                const std::string &suffix="") const {
    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler(this->dof_handler);
    const int num_groups = flux.n_blocks();
    const int digits = std::to_string(num_groups).size();
    AssertDimension(group_structure.size(), num_groups+1);
    dealii::BlockVector<double> flux_m(num_groups, this->dof_handler.n_dofs());
    for (int g = 0; g < num_groups; ++g) {
      const std::string gs = std::to_string(g+1);
      d2m.vmult(flux_m.block(g), flux.block(g));
      if (!group_structure.empty()) {
        int g_rev = num_groups - 1 - g;
        flux_m.block(g) /=
            std::log(group_structure[g_rev+1] / group_structure[g_rev]);
      }
      data_out.add_data_vector(
          flux_m.block(g), "g" + std::string(digits - gs.size(), '0') + gs,
          dealii::DataOut_DoFData<dealii::DoFHandler<dim>,
                                  dim>::DataVectorType::type_dof_data);
    }
    data_out.build_patches();
    std::string name = GetTestName();
    if (!suffix.empty())
      name += "-" + suffix;
    std::ofstream output(name+".vtu");
    data_out.write_vtu(output);
  }

  void PlotDiffAngular(const dealii::BlockVector<double> &flux,
                       const dealii::BlockVector<double> &ref,
                       const DiscreteToMoment<dim, qdim> &d2m,
                       const std::string &suffix="") {
    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler(this->dof_handler);
    const int num_groups = flux.n_blocks();
    const int digits = std::to_string(num_groups).size();
    dealii::BlockVector<double> diff_m(num_groups, this->dof_handler.n_dofs());
    dealii::Vector<double> diff_g(
        this->quadrature.size()*this->dof_handler.n_dofs());
    dealii::Vector<double> ref_m(this->dof_handler.n_dofs());
    for (int g = 0; g < num_groups; ++g) {
      const std::string gs = std::to_string(g+1);
      diff_g = flux.block(g);
      diff_g -= ref.block(g);
      d2m.vmult(diff_m.block(g), diff_g);
      d2m.vmult(ref_m, ref.block(g));
      for (int i = 0; i < diff_m.block(g).size(); ++i)
        diff_m.block(g)[i] /= ref_m[i];
      data_out.add_data_vector(
          diff_m.block(g), "g" + std::string(digits - gs.size(), '0') + gs,
          dealii::DataOut_DoFData<dealii::DoFHandler<dim>,
                                  dim>::DataVectorType::type_dof_data);
    }
    data_out.build_patches();
    std::string name = GetTestName();
    if (!suffix.empty())
      name += "-" + suffix;
    std::ofstream output(name+".vtu");
    data_out.write_vtu(output);
  }

  void PlotDiffScalar(const dealii::BlockVector<double> &flux,
                      const dealii::BlockVector<double> &ref,
                      const DiscreteToMoment<dim, qdim> &d2m,
                      const std::string &suffix="") {
    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler(this->dof_handler);
    const int num_groups = flux.n_blocks();
    const int digits = std::to_string(num_groups).size();
    dealii::BlockVector<double> diff_m(num_groups, this->dof_handler.n_dofs());
    dealii::Vector<double> ref_g(this->dof_handler.n_dofs());
    for (int g = 0; g < num_groups; ++g) {
      const std::string gs = std::to_string(g+1);
      d2m.vmult(ref_g, ref.block(g));
      d2m.vmult(diff_m.block(g), flux.block(g));
      diff_m.block(g) -= ref_g;
      for (int i = 0; i < diff_m.block(g).size(); ++i)
        diff_m.block(g)[i] /= ref_g[i];
      data_out.add_data_vector(
          diff_m.block(g), "g" + std::string(digits - gs.size(), '0') + gs,
          dealii::DataOut_DoFData<dealii::DoFHandler<dim>,
                                  dim>::DataVectorType::type_dof_data);
    }
    data_out.build_patches();
    std::string name = GetTestName();
    if (!suffix.empty())
      name += "-" + suffix;
    std::ofstream output(name+".vtu");
    data_out.write_vtu(output);
  }
};

#endif  // AETHER_EXAMPLES_EXAMPLE_TEST_H_