#ifndef AETHER_BASE_MGXS_H_
#define AETHER_BASE_MGXS_H_

#include <deal.II/base/hdf5.h>

namespace aether {

template <int num_groups, int num_mats, int num_ell>
struct Mgxs {
  std::array<std::array<double, num_mats>, num_groups> total;
  std::array<std::array<double, num_mats>, num_groups> chi;
  std::array<std::array<double, num_mats>, num_groups> nu_fission;
  std::array<std::array<std::array<std::array<
    double, num_ell>, num_mats>, num_groups>, num_groups> scatter;
};

template <int num_groups, int num_mats, int num_ell>
Mgxs<num_groups, num_mats, num_ell> read_mgxs(
    std::string filename, 
    std::string temperature,
    std::vector<std::string> materials) {
  namespace HDF5 = dealii::HDF5;
  HDF5::File file(filename, HDF5::File::FileAccessMode::open);
  AssertDimension(num_groups, file.get_attribute<int>("energy_groups"));
  AssertDimension(num_mats, materials.size());
  AssertDimension(num_ell, 1);
  std::vector<std::vector<double>> total_pivot(
    num_mats, std::vector<double>(num_groups));
  auto chi_pivot = total_pivot;
  auto nu_fission_pivot = total_pivot;
  Mgxs<num_groups, num_mats, num_ell> mgxs{};
  for (int j = 0; j < num_mats; ++j) {
    HDF5::Group material = file.open_group(materials[j]);
    int order = material.get_attribute<int>("order");
    Assert(order < num_ell, dealii::ExcInvalidState());
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
          mgxs.scatter[g][gp][j][ell] = scatter_matrix[gg];
        }
      }
    }
    AssertDimension(gg, scatter_matrix.size());
  }
  // Pivot cross-section data
  for (int g = 0; g < num_groups; ++g) {
    for (int j = 0; j < num_mats; ++j) {
      mgxs.total[g][j] = total_pivot[j][g];
      mgxs.chi[g][j] = chi_pivot[j][g];
      mgxs.nu_fission[g][j] = nu_fission_pivot[j][g];
    }
  }
  return mgxs;
}

}

#endif  // AETHER_BASE_MGXS_H_