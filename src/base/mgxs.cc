#include "base/mgxs.h"

namespace aether {

Mgxs read_mgxs(
    const std::string &filename, 
    const std::string &temperature,
    const std::vector<std::string> &materials,
    const bool read_structure) {
  namespace HDF5 = dealii::HDF5;
  HDF5::File file(filename, HDF5::File::FileAccessMode::open);
  const int num_groups =  file.get_attribute<int>("energy_groups");
  const int num_materials = materials.size();
  const int num_legendre = 1;
  Mgxs mgxs(num_groups, num_materials, num_legendre);
  read_mgxs(mgxs, filename, temperature, materials, read_structure);
  return mgxs;
}

void read_mgxs(
    Mgxs &mgxs,
    const std::string &filename, 
    const std::string &temperature,
    const std::vector<std::string> &materials,
    const bool read_structure) {
  namespace HDF5 = dealii::HDF5;
  HDF5::File file(filename, HDF5::File::FileAccessMode::open);
  const int num_groups =  file.get_attribute<int>("energy_groups");
  const int num_materials = materials.size();
  const int num_legendre = 1;
  if (read_structure)
    mgxs.group_structure =
        file.open_dataset("group structure").read<std::vector<double>>();
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
    std::vector<double> chi(num_groups);
    std::vector<double> nu_fission(num_groups);
    std::vector<double> scatter_matrix(num_groups * num_groups);
    bool is_fissionable = false;
    for (int g = 0; g < num_groups; ++g) {
      total[g] = mgxs.total[g][j];
      chi[g] = mgxs.chi[g][j];
      nu_fission[g] = mgxs.nu_fission[g][j];
      if (chi[g] > 0 || nu_fission[g] > 0)
        is_fissionable = true;
      for (int gp = 0; gp < num_groups; ++gp) {
        scatter_matrix[g*num_groups + gp] = mgxs.scatter[gp][g][j];
      }
    }
    library.write_dataset("total", total);
    if (is_fissionable) {
      library.write_dataset("chi", chi);
      library.write_dataset("nu-fission", nu_fission);
    }
    material.set_attribute<int>("fissionable", is_fissionable);
    HDF5::Group scatter_data = library.create_group("scatter_data");
    scatter_data.write_dataset("g_min", std::vector<int>(num_groups, 1));
    scatter_data.write_dataset("g_max", std::vector<int>(num_groups, num_groups));
    scatter_data.write_dataset("scatter_matrix", scatter_matrix);
  }
  std::stringstream datetime;
  std::time_t t = std::time(nullptr);
  datetime << std::put_time(std::localtime(&t), "%F %T");
  file.set_attribute("datetime", datetime.str());
}

template <int dim, int qdim = dim == 1 ? 1 : 2>
void collapse_spectra(std::vector<dealii::BlockVector<double>> &spectra,
                      const dealii::BlockVector<double> &flux,
                      const dealii::DoFHandler<dim> &dof_handler,
                      const sn::Transport<dim, qdim> &transport) {
  AssertThrow(spectra.empty(), dealii::ExcInvalidState());
  AssertThrow(flux.block(0).size() % dof_handler.n_dofs() == 0,
      dealii::ExcNotMultiple(flux.block(0).size(), dof_handler.n_dofs()));
  const int num_groups = flux.n_blocks();
  const int order = (flux.block(0).size() / dof_handler.n_dofs()) - 1;
  spectra.resize(1, dealii::BlockVector<double>(order+1, num_groups));
  std::vector<dealii::types::global_dof_index> dof_indices(
      dof_handler.get_fe().dofs_per_cell);
  for (int g = 0; g < num_groups; ++g) {
    dealii::BlockVector<double> flux_g(order+1, dof_handler.n_dofs());
    flux_g = flux.block(g);
    int c = 0;
    for (auto cell = dof_handler.begin_active(); cell != dof_handler.end(); 
         ++cell) {
      if (!cell->is_locally_owned())
        continue;
      ++c;
      cell->get_dof_indices(dof_indices);
      const int material = cell->material_id();
      if (material+1 > spectra.size())
        spectra.resize(material+1, 
                       dealii::BlockVector<double>(order+1, num_groups));
      const dealii::FullMatrix<double> &mass = transport.cell_matrices[c].mass;
      for (int ell = 0; ell <= order; ++ell)
        for (int i = 0; i < mass.n(); ++i)
          for (int j = 0; j < mass.m(); ++j)
            spectra[material].block(ell)[g] += 
                mass[i][j] * flux_g.block(ell)[dof_indices[j]];
    }
  }
}

template <int dim, int qdim = dim == 1 ? 1 : 2>
Mgxs collapse_mgxs(const dealii::BlockVector<double> &flux,
                   const dealii::DoFHandler<dim> &dof_handler,
                   const sn::Transport<dim, qdim> &transport,
                   const Mgxs &mgxs,
                   const std::vector<int> &g_maxes,
                   const TransportCorrection correction) {
  std::vector<dealii::BlockVector<double>> spectra;
  // std::cout << "collapse spectra\n";
  collapse_spectra(spectra, flux, dof_handler, transport);
  AssertDimension(spectra.size(), mgxs.total.size());
  AssertDimension(spectra[0].size(), mgxs.total[0].size());
  // std::cout << "collapse mgxs\n";
  return collapse_mgxs(spectra, mgxs, g_maxes, correction);
}

Mgxs collapse_mgxs(const dealii::Vector<double> &spectrum,
                   const Mgxs &mgxs, const std::vector<int> &g_maxes,
                   const TransportCorrection correction) {
  const int num_materials = mgxs.total[0].size();
  dealii::BlockVector<double> spectrum_b(1, spectrum.size());
  spectrum_b = spectrum;
  std::vector<dealii::BlockVector<double>> spectra(num_materials, spectrum_b);
  Assert(correction == CONSISTENT_P, dealii::ExcInvalidState());
  return collapse_mgxs(spectra, mgxs, g_maxes, correction);
}

Mgxs collapse_mgxs(const std::vector<dealii::BlockVector<double>> &spectra,
                   const Mgxs &mgxs, const std::vector<int> &g_maxes,
                   const TransportCorrection correction) {
  const int order = spectra[0].n_blocks() - 1;
  const int order_coarse = order - (correction == CONSISTENT_P ? 0 : 1);
  const int num_materials = mgxs.total[0].size();
  const int num_groups_coarse = g_maxes.size();
  Mgxs mgxs_coarse(num_groups_coarse, num_materials, order_coarse+1);
  const int length = num_materials * (order + 1);
  for (int g_coarse = 0; g_coarse < num_groups_coarse; ++g_coarse) {
    int g_min = g_coarse == 0 ? 0 : g_maxes[g_coarse-1];
    int g_max = g_maxes[g_coarse];
    std::vector<double> denominator(length);
    std::vector<double> collision(length);
    std::vector<std::vector<double>> scattering(num_groups_coarse,
        std::vector<double>(length));
    for (int g = g_min; g < g_max; ++g) {
      for (int j = 0; j < num_materials; ++j) {
        for (int ell = 0; ell <= order; ++ell) {
          int jl = j + ell * num_materials;
          collision[jl] += mgxs.total[g][j] * spectra[j].block(ell)[g];
          denominator[jl] += spectra[j].block(ell)[g];
          if (correction == INCONSISTENT_P && ell == order)
            continue;  // don't need L+1 scattering moments, only collision
          // For convenience, the usual meanings of g and g' (gp) are reversed
          // That is to say, neutrons scatter from g to g'
          for (int gp_coarse = 0; gp_coarse < num_groups_coarse; ++gp_coarse) {
            int gp_min = gp_coarse == 0 ? 0 : g_maxes[gp_coarse-1];
            int gp_max = g_maxes[gp_coarse];
            for (int gp = gp_min; gp < gp_max; ++gp) {
              scattering[gp_coarse][jl] += mgxs.scatter[gp][g][jl] 
                                           * spectra[j].block(ell)[g];
            }
          }
        }
      }
    }
    for (int j = 0; j < num_materials; ++j) {
      switch (correction) {
        case CONSISTENT_P:
          mgxs_coarse.total[g_coarse][j] = collision[j] / denominator[j]; 
          break;
        case INCONSISTENT_P: {
          int jll = j + order * num_materials;
        // std::cout << "g=" << g_coarse << " ell=" << order << " " 
        //           << denominator[jll] << std::endl;
          mgxs_coarse.total[g_coarse][j] = collision[jll] / denominator[jll];
          break;
        }
        default:
          throw dealii::ExcNotImplemented();
      }
      for (int ell = 0; ell <= order_coarse; ++ell) {
        int jl = j + ell * num_materials;
        // std::cout << "g=" << g_coarse << " ell=" << ell << " " 
        //           << denominator[jl] << std::endl;
        for (int gp_coarse = 0; gp_coarse < num_groups_coarse; ++gp_coarse) {
          mgxs_coarse.scatter[gp_coarse][g_coarse][jl] = 
              scattering[gp_coarse][jl] / denominator[jl];
          if (gp_coarse == g_coarse)
            mgxs_coarse.scatter[gp_coarse][g_coarse][jl] -=
                collision[jl]/denominator[jl] - mgxs_coarse.total[g_coarse][j];
        }
      }
    }
  }
  return mgxs_coarse;
}


template void collapse_spectra<1>(std::vector<dealii::BlockVector<double>>&,
                                  const dealii::BlockVector<double>&,
                                  const dealii::DoFHandler<1>&,
                                  const sn::Transport<1, 1>&);
template void collapse_spectra<2>(std::vector<dealii::BlockVector<double>>&,
                                  const dealii::BlockVector<double>&,
                                  const dealii::DoFHandler<2>&,
                                  const sn::Transport<2, 2>&);
template void collapse_spectra<3>(std::vector<dealii::BlockVector<double>>&,
                                  const dealii::BlockVector<double>&,
                                  const dealii::DoFHandler<3>&,
                                  const sn::Transport<3, 2>&);

template Mgxs collapse_mgxs<1>(const dealii::BlockVector<double>&,
                               const dealii::DoFHandler<1>&,
                               const sn::Transport<1, 1>&,
                               const Mgxs&,
                               const std::vector<int>&,
                               const TransportCorrection);
template Mgxs collapse_mgxs<2>(const dealii::BlockVector<double>&,
                               const dealii::DoFHandler<2>&,
                               const sn::Transport<2, 2>&,
                               const Mgxs&,
                               const std::vector<int>&,
                               const TransportCorrection);
template Mgxs collapse_mgxs<3>(const dealii::BlockVector<double>&,
                               const dealii::DoFHandler<3>&,
                               const sn::Transport<3, 2>&,
                               const Mgxs&,
                               const std::vector<int>&,
                               const TransportCorrection);

}  // namespace aether