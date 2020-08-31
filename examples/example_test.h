#ifndef AETHER_EXAMPLES_EXAMPLE_TEST_H_
#define AETHER_EXAMPLES_EXAMPLE_TEST_H_

#include <regex>
#include <hdf5.h>
#include <ctime>

#include <deal.II/base/hdf5.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
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
  QPglc<qdim> quadrature;
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

  void WriteFlux(const dealii::BlockVector<double> &flux,
                 const std::vector<double> &history,
                 const std::string &filename) {
    dealii::Vector<double> flux_v(flux.size());
    flux_v = flux;
    namespace HDF5 = dealii::HDF5;
    HDF5::File file(filename, HDF5::File::FileAccessMode::create);
    file.write_dataset("flux_full", flux_v);
    file.write_dataset("history_data", history);
    // add metadata
    file.set_attribute("n_dofs", this->dof_handler.n_dofs());
    file.set_attribute("n_polar", this->quadrature.get_tensor_basis()[0].size());
    if (qdim == 2)
      file.set_attribute("n_azim", this->quadrature.get_tensor_basis()[1].size());
    file.set_attribute("n_groups", int(this->mgxs->total.size()));
    // timestamp
    std::stringstream datetime;
    std::time_t t = std::time(nullptr);
    datetime << std::put_time(std::localtime(&t), "%F %T");
    const std::string datetime_str = datetime.str();
    file.set_attribute("datetime", datetime_str);
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
    // dealii::BlockVector<double> flux_m(num_groups, 3*this->dof_handler.n_dofs());
    std::vector<dealii::BlockVector<double>> moments(num_groups,
        dealii::BlockVector<double>(d2m.n_block_rows(1), this->dof_handler.n_dofs()));
    dealii::Vector<double> flat(moments[0].size());
    for (int g = 0; g < num_groups; ++g) {
      std::string gs = std::to_string(g+1);
      gs = "g" + std::string(digits - gs.size(), '0') + gs + "_";
      d2m.vmult(flat, flux.block(g));
      if (!group_structure.empty()) {
        int g_rev = num_groups - 1 - g;
        flat /=
            std::log(group_structure[g_rev+1] / group_structure[g_rev]);
      }
      moments[g] = flat;
      // moments[g].block(1) += moments[g].block(2);
      data_out.add_data_vector(moments[g].block(0), gs+"scalar");
      data_out.add_data_vector(moments[g].block(1), gs+"current_y");
      data_out.add_data_vector(moments[g].block(2), gs+"current_x");
      // data_out.add_data_vector(
      //     moments[g], {gs+"scalar", gs+"current_x", gs+"current_y"},
      //     dealii::DataOut_DoFData<dealii::DoFHandler<dim>, dim>::DataVectorType::type_dof_data,
      //     {dealii::DataComponentInterpretation::component_is_scalar,
      //         dealii::DataComponentInterpretation::component_is_scalar,
      //         dealii::DataComponentInterpretation::component_is_scalar});
    }
    data_out.build_patches();
    std::string name = GetTestName();
    if (!suffix.empty())
      name += "-" + suffix;
    std::ofstream output(name+".vtu");
    data_out.write_vtu(output);
    /*
    // plot current
    dealii::FESystem<dim, dim> fe(this->dof_handler.get_fe(), dim);
    dealii::DoFHandler<dim, dim> dof_handler_v;
    dof_handler_v.initialize(
        this->dof_handler.get_triangulation(), fe);
    dealii::DataOut<dim> data_out_v;
    data_out_v.attach_dof_handler(dof_handler_v);
    for (int g = 0; g < moments.size(); ++g) {
      std::string gs = std::to_string(g+1);
      gs = "g" + std::string(digits - gs.size(), '0') + gs + "_";
      data_out_v.add_data_vector(
          moments[g], {gs+"scalar", gs+"current", gs+"current"},
          dealii::DataOut_DoFData<dealii::DoFHandler<dim>, dim>::DataVectorType::type_dof_data,
          {dealii::DataComponentInterpretation::component_is_scalar,
           dealii::DataComponentInterpretation::component_is_part_of_vector,
           dealii::DataComponentInterpretation::component_is_part_of_vector});
    }
    data_out_v.build_patches();
    std::ofstream output_v(name+"_v.vtu");
    data_out_v.write_vtu(output_v);
    */
  }

  void PlotDiffAngular(const dealii::BlockVector<double> &flux,
                       const dealii::BlockVector<double> &ref,
                       const DiscreteToMoment<dim, qdim> &d2m,
                       const std::string &suffix="") {
    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler(this->dof_handler);
    const int num_groups = flux.n_blocks();
    const int digits = std::to_string(num_groups).size();
    // dealii::BlockVector<double> diff_m(num_groups, this->dof_handler.n_dofs());
    std::vector<dealii::BlockVector<double>> diff_m(
        num_groups, dealii::BlockVector<double>(
          d2m.n_block_rows(1), this->dof_handler.n_dofs()));
    dealii::Vector<double> flat(diff_m[0].size());
    dealii::Vector<double> diff_g(
        this->quadrature.size()*this->dof_handler.n_dofs());
    dealii::Vector<double> ref_m(this->dof_handler.n_dofs());
    for (int g = 0; g < num_groups; ++g) {
      std::string gs = std::to_string(g+1);
      gs = "g" + std::string(digits - gs.size(), '0') + gs + "_";
      diff_g = flux.block(g);
      diff_g -= ref.block(g);
      d2m.vmult(flat, diff_g);
      d2m.vmult(ref_m, ref.block(g));
      diff_m[g] = flat;
      for (int lm = 0; lm < diff_m[g].n_blocks(); ++lm)
        for (int i = 0; i < diff_m[g].block(lm).size(); ++i)
          diff_m[g].block(lm)[i] /= ref_m[i];
      data_out.add_data_vector(diff_m[g].block(0), gs+"scalar");
      data_out.add_data_vector(diff_m[g].block(1), gs+"current_y");
      data_out.add_data_vector(diff_m[g].block(2), gs+"current_x");
      // data_out.add_data_vector(
      //     diff_m.block(g), "g" + std::string(digits - gs.size(), '0') + gs,
      //     dealii::DataOut_DoFData<dealii::DoFHandler<dim>,
      //                             dim>::DataVectorType::type_dof_data);
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