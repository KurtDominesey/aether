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
  std::vector<std::vector<double>> total_pivot(
    num_materials, std::vector<double>(num_groups));
  auto chi_pivot = total_pivot;
  auto nu_fission_pivot = total_pivot;
  Mgxs mgxs(num_groups, num_materials, num_legendre);
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
  return mgxs;
}

}  // namespace aether