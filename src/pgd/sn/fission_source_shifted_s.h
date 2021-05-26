#ifndef AETHER_PGD_SN_FISSION_SOURCE_SHIFTED_S_H_
#define AETHER_PGD_SN_FISSION_SOURCE_SHIFTED_S_H_

#include <deal.II/lac/block_vector.h>

#include "pgd/sn/fission_s.h"
#include "pgd/sn/fixed_source_s.h"
#include "pgd/sn/fixed_source_s_gs.h"

namespace aether::pgd::sn {

template <int dim, int qdim = dim == 1 ? 1 : 2>
class FissionSourceShiftedS {
 public:
  FissionSourceShiftedS(const FissionS<dim, qdim> &fission_s, 
                        const FixedSourceS<dim, qdim> &fixed_source_s,
                        const FixedSourceSGS<dim, qdim> &fixed_source_s_gs);
  void vmult(dealii::BlockVector<double> &dst,
             const dealii::BlockVector<double> &src) const;
  double shift = 1;

 protected:
  const FissionS<dim, qdim> &fission_s;
  const FixedSourceS<dim, qdim> &fixed_source_s;
  const FixedSourceSGS<dim, qdim> &fixed_source_s_gs;
};

}  // namespace aether::pgd::sn

#endif  // AETHER_PGD_SN_FISSION_SOURCE_SHIFTED_S_H_