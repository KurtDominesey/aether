#include "base/mgxs.h"

namespace aether {

Mgxs read_mgxs(
    const std::string &filename, 
    const std::string &temperature,
    const std::vector<std::string> &materials) {
  namespace HDF5 = dealii::HDF5;
  HDF5::File file(filename, HDF5::File::FileAccessMode::open);
  const int num_groups =  file.get_attribute<int>("energy_groups");
  const int num_materials = materials.size();
  const int num_legendre = 1;
  Mgxs mgxs(num_groups, num_materials, num_legendre);
  read_mgxs(mgxs, filename, temperature, materials);
  return mgxs;
}

void read_mgxs(
    Mgxs &mgxs,
    const std::string &filename, 
    const std::string &temperature,
    const std::vector<std::string> &materials) {
  namespace HDF5 = dealii::HDF5;
  HDF5::File file(filename, HDF5::File::FileAccessMode::open);
  const int num_groups =  file.get_attribute<int>("energy_groups");
  const int num_materials = materials.size();
  const int num_legendre = 1;
  std::vector<std::vector<double>> total_pivot(
    num_materials, std::vector<double>(num_groups));
  auto chi_pivot = total_pivot;
  auto nu_fission_pivot = total_pivot;
  for (int j = 0; j < num_materials; ++j) {
    if (materials[j] == "void")
      continue;
    HDF5::Group material = file.open_group(materials[j]);
    int order = material.get_attribute<int>("order");
    Assert(order < num_legendre, dealii::ExcInvalidState());
    // We should check scatter_format is "legendre", but OpenMC encodes MGXS
    // files with ASCII strings, whereas dealii::HDF5 can only read UTF8.
    HDF5::Group library = material.open_group(temperature);
    total_pivot[j] = library.open_dataset("total").read<std::vector<double>>();
    if (material.get_attribute<bool>("fissionable")) {
      chi_pivot[j] = 
          library.open_dataset("chi").read<std::vector<double>>();
      nu_fission_pivot[j] = 
          library.open_dataset("nu-fission").read<std::vector<double>>();
    }
    HDF5::Group scatter_data = library.open_group("scatter_data");
    std::vector<int> g_min = 
        scatter_data.open_dataset("g_min").read<std::vector<int>>();
    std::vector<int> g_max = 
        scatter_data.open_dataset("g_max").read<std::vector<int>>();
    std::vector<double> scatter_matrix =
        scatter_data.open_dataset("scatter_matrix").read<std::vector<double>>();
    // Also cannot read scatter_shape. Assume [G][G'][Order].
    // Aether uses opposite notation of OpenMC, in that g (G) denotes outgoing
    // groups, while g' (gp, G') is incoming.
    int gg = 0;
    for (int ga = 0; ga < num_groups; ++ga) {
      for (int gb = g_min[ga]; gb <= g_max[ga]; ++gb) {
        for (int ell = 0; ell <= order; ++ell, ++gg) {
          int gp = ga;
          int g = gb - 1;
          mgxs.scatter[g][gp][j*num_legendre + ell] = scatter_matrix[gg];
        }
      }
    }
    AssertDimension(gg, scatter_matrix.size());
  }
  // Pivot cross-section data
  for (int g = 0; g < num_groups; ++g) {
    for (int j = 0; j < num_materials; ++j) {
      mgxs.total[g][j] = total_pivot[j][g];
      mgxs.chi[g][j] = chi_pivot[j][g];
      mgxs.nu_fission[g][j] = nu_fission_pivot[j][g];
    }
  }
}

void write_mgxs(
    const Mgxs& mgxs,
    const std::string &filename, 
    const std::string &temperature,
    const std::vector<std::string> &materials) {
  namespace HDF5 = dealii::HDF5;
  HDF5::File file(filename, HDF5::File::FileAccessMode::create);
  const int num_groups = mgxs.total.size();
  const int num_materials = materials.size();
  const int num_legendre = 1;
  file.set_attribute<int>("energy_groups", num_groups);
  for (int j = 0; j < num_materials; ++j) {
    HDF5::Group material = file.create_group(materials[j]);
    material.set_attribute<int>("order", num_legendre - 1);
    HDF5::Group library = material.create_group(temperature);
    std::vector<double> total(num_groups);
    std::vector<double> scatter_matrix(num_groups * num_groups);
    for (int g = 0; g < num_groups; ++g) {
      total[g] = mgxs.total[g][j];
      for (int gp = 0; gp < num_groups; ++gp) {
        scatter_matrix[g*num_groups + gp] = mgxs.scatter[gp][g][j];
      }
    }
    library.write_dataset("total", total);
    HDF5::Group scatter_data = library.create_group("scatter_data");
    scatter_data.write_dataset("g_max", std::vector<int>(num_groups, 1));
    scatter_data.write_dataset("g_min", std::vector<int>(num_groups, num_groups));
    scatter_data.write_dataset("scatter_matrix", scatter_matrix);
  }
}

template <int dim, int qdim = dim == 1 ? 1 : 2>
Mgxs collapse_mgxs(const dealii::BlockVector<double> &flux,
                   const dealii::DoFHandler<dim> &dof_handler,
                   const sn::Transport<dim, qdim> &transport,
                   const Mgxs &mgxs,
                   const std::vector<int> &g_maxes) {
  const int num_groups = mgxs.total.size();
  const int num_materials = mgxs.total[0].size();
  std::vector<std::vector<double>> integrals(num_groups,
      std::vector<double>(num_materials));
  std::vector<dealii::types::global_dof_index> dof_indices(
      dof_handler.get_fe().dofs_per_cell);
  for (int g = 0; g < num_groups; ++g) {
    dealii::BlockVector<double> flux_g(1, dof_handler.n_dofs());
    flux_g = flux.block(g);
    int c = 0;
    for (auto cell = dof_handler.begin_active(); cell != dof_handler.end(); 
         ++cell) {
      if (!cell->is_locally_owned())
        continue;
      ++c;
      cell->get_dof_indices(dof_indices);
      const int material = cell->material_id();
      const dealii::FullMatrix<double> &mass = transport.cell_matrices[c].mass;
      for (int i = 0; i < mass.n(); ++i)
        for (int j = 0; j < mass.m(); ++j)
          integrals[g][material] += mass[i][j]*flux_g.block(0)[dof_indices[j]];
    }
  }
  const int num_groups_coarse = g_maxes.size();
  Mgxs mgxs_coarse(num_groups_coarse, num_materials, 1);
  int g_min = 0;
  int g_max = 0;
  for (int g_coarse = 0; g_coarse < num_groups_coarse; ++g_coarse) {
    g_min = g_max;
    g_max = g_maxes[g_coarse];
    std::vector<double> collision(num_materials);
    std::vector<double> denominator(num_materials);
    std::vector<std::vector<double>> scattering(num_groups_coarse,
        std::vector<double>(num_materials));
    for (int g = g_min; g < g_max; ++g) {
      for (int j = 0; j < num_materials; ++j) {
        collision[j] += mgxs.total[g][j] * integrals[g][j];
        denominator[j] += integrals[g][j];
        // For convenience, the usual meanings of g and g' (gp) are reversed
        // That is to say, neutrons scatter from g to g'
        int gp_min = 0;
        int gp_max = 0;
        for (int gp_coarse = 0; gp_coarse < num_groups_coarse; ++gp_coarse) {
          int gp_min = gp_max;
          int gp_max = g_maxes[gp_coarse];
          for (int gp = gp_min; gp < gp_max; ++gp) {
            scattering[gp_coarse][j] += mgxs.scatter[g][gp][j]*integrals[g][j];
          }
        }
      }
    }
    for (int j = 0; j < num_materials; ++j) {
      mgxs_coarse.total[g_coarse][j] = collision[j] / denominator[j];
      for (int gp_coarse = 0; gp_coarse < num_groups_coarse; ++gp_coarse) {
        mgxs_coarse.scatter[g_coarse][gp_coarse][j] = 
            scattering[gp_coarse][j] / denominator[j];
      }
    }
  }
  return mgxs_coarse;
}

template Mgxs collapse_mgxs<1>(const dealii::BlockVector<double>&,
                               const dealii::DoFHandler<1>&,
                               const sn::Transport<1, 1>&,
                               const Mgxs&,
                               const std::vector<int>&);
template Mgxs collapse_mgxs<2>(const dealii::BlockVector<double>&,
                               const dealii::DoFHandler<2>&,
                               const sn::Transport<2, 2>&,
                               const Mgxs&,
                               const std::vector<int>&);
template Mgxs collapse_mgxs<3>(const dealii::BlockVector<double>&,
                               const dealii::DoFHandler<3>&,
                               const sn::Transport<3, 2>&,
                               const Mgxs&,
                               const std::vector<int>&);

}  // namespace aether