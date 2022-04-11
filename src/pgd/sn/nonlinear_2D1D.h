#ifndef AETHER_PGD_SN_NONLINEAR_2D1D_H_
#define AETHER_PGD_SN_NONLINEAR_2D1D_H_

#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/vector.h>

#include "pgd/sn/inner_products.h"
#include "pgd/sn/fixed_source_2D1D.h"

namespace aether::pgd::sn {

class Nonlinear2D1D {
 public:
  Nonlinear2D1D(FixedSource2D1D<1> &one_d, FixedSource2D1D<2> &two_d, 
                const std::vector<std::vector<int>> &materials,
                const Mgxs &mgxs,
                bool both_mg);
  void enrich();
  double iter();
  void reweight();
 protected:
  FixedSource2D1D<1> &one_d;
  FixedSource2D1D<2> &two_d;
  const Mgxs &mgxs;
  const std::vector<std::vector<int>> &materials;
  bool both_mg;
  dealii::LAPACKFullMatrix<double> matrix_wgt;
  dealii::Vector<double> weights;
  dealii::Vector<double> src_wgt;
};

}  // namespace aether::pgd::sn

#endif  // AETHER_PGD_SN_NONLINEAR_2D1D_H_