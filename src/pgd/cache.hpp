#ifndef AETHER_PGD_CACHE_H_
#define AETHER_PGD_CACHE_H_

#include <deal.II/lac/block_vector.h>

struct Cache {
  template <typename T>
  using Separated = std::map<std::string, std::vector<T> >;

  std::vector<dealii::BlockVector<double> > modes;
  std::vector<Separated<dealii::BlockVector<double> > > operated_modes;
  bool modes_updated = false;
};

#endif // AETHER_PGD_CACHE_H_