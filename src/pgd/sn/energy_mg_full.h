#ifndef AETHER_PGD_SN_ENERGY_MG_FULL_H_
#define AETHER_PGD_SN_ENERGY_MG_FULL_H_

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

#include "pgd/sn/inner_products.h"
#include "pgd/sn/linear_interface.h"
#include "base/mgxs.h"

namespace aether::pgd::sn {

class EnergyMgFull : public LinearInterface {
 public:
  EnergyMgFull(const Mgxs &mgxs,
               const std::vector<dealii::Vector<double>> &sources);
  void vmult(dealii::BlockVector<double> &dst,
             const dealii::BlockVector<double> &src,
             std::vector<InnerProducts> coefficients_x,
             std::vector<double> coefficients_b);
  void step(dealii::BlockVector<double> &x,
            const dealii::BlockVector<double> &b,
            std::vector<InnerProducts> coefficients_x,
            std::vector<double> coefficients_b);
  void get_inner_products(std::vector<InnerProducts> &inner_products_x,
                          std::vector<double> &inner_products_b);
  void enrich();
  std::vector<dealii::Vector<double>> modes;
 protected:
  const Mgxs &mgxs;
  const std::vector<dealii::Vector<double>> &sources;
  dealii::FullMatrix<double> matrix;
  dealii::Vector<double> source;
  void set_matrix(InnerProducts coefficients_x);
  void set_source(std::vector<double> coefficients_b, 
                  std::vector<InnerProducts> coefficients_x);
};

}

#endif  // AETHER_PGD_SN_ENERGY_MG_FULL_H_