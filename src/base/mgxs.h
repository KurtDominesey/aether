#ifndef AETHER_BASE_MGXS_H_
#define AETHER_BASE_MGXS_H_

#include <hdf5.h>

#include <deal.II/base/hdf5.h>
#include <deal.II/dofs/dof_handler.h>

#include "sn/transport.h"

namespace aether {

namespace sn {
template <int dim, int qdim> class Transport;
}

struct Mgxs {
  Mgxs(int num_groups, int num_materials, int num_legendre)
      : group_structure(num_groups + 1),
        total(num_groups, std::vector<double>(num_materials)),
        chi(num_groups, std::vector<double>(num_materials)),
        nu_fission(num_groups, std::vector<double>(num_materials)),
        scatter(num_groups, std::vector<std::vector<double>>(
                num_groups, std::vector<double>(num_materials*num_legendre))) {}
  std::vector<double> group_structure;
  std::vector<std::vector<double>> total;
  std::vector<std::vector<double>> chi;
  std::vector<std::vector<double>> nu_fission;
  std::vector<std::vector<std::vector<double>>> scatter;
};

Mgxs read_mgxs(const std::string &filename, 
               const std::string &temperature,
               const std::vector<std::string> &materials);

void read_mgxs(Mgxs &mgxs,
               const std::string &filename, 
               const std::string &temperature,
               const std::vector<std::string> &materials);

void write_mgxs(const Mgxs& mgxs,
                const std::string &filename, 
                const std::string &temperature,
                const std::vector<std::string> &materials);

template <int dim, int qdim = dim == 1 ? 1 : 2>
void collapse_spectra(std::vector<dealii::Vector<double>> &spectra,
                      const dealii::BlockVector<double> &flux,
                      const dealii::DoFHandler<dim> &dof_handler,
                      const sn::Transport<dim, qdim> &transport);

template <int dim, int qdim = dim == 1 ? 1 : 2>
Mgxs collapse_mgxs(const dealii::BlockVector<double> &flux,
                   const dealii::DoFHandler<dim> &dof_handler,
                   const sn::Transport<dim, qdim> &transport,
                   const Mgxs &mgxs,
                   const std::vector<int> &g_maxes);

Mgxs collapse_mgxs(const dealii::Vector<double> &spectrum, 
                   const Mgxs &mgxs, const std::vector<int> &g_maxes);

Mgxs collapse_mgxs(const std::vector<dealii::Vector<double>> &spectra,
                   const Mgxs &mgxs, const std::vector<int> &g_maxes);

}

#endif  // AETHER_BASE_MGXS_H_