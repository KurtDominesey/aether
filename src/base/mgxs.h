#ifndef AETHER_BASE_MGXS_H_
#define AETHER_BASE_MGXS_H_

#include <deal.II/base/hdf5.h>

namespace aether {

struct Mgxs {
  Mgxs(int num_groups, int num_materials, int num_legendre)
      : total(num_groups, std::vector<double>(num_materials)),
        chi(num_groups, std::vector<double>(num_materials)),
        nu_fission(num_groups, std::vector<double>(num_materials)),
        scatter(num_groups, std::vector<std::vector<double>>(
                num_groups, std::vector<double>(num_materials*num_legendre))) {}
  std::vector<std::vector<double>> total;
  std::vector<std::vector<double>> chi;
  std::vector<std::vector<double>> nu_fission;
  std::vector<std::vector<std::vector<double>>> scatter;
};

Mgxs read_mgxs(const std::string &filename, 
               const std::string &temperature,
               const std::vector<std::string> &materials);

}

#endif  // AETHER_BASE_MGXS_H_