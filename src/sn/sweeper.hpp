#ifndef AETHER_SN_SWEEPER_H_
#define AETHER_SN_SWEEPER_H_

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_vector.h>

class Sweeper {
 public:
  void initialize(std::vector<int> &octant_to_global,
                  dealii::BlockVector<double> &dst_ref, 
                  dealii::BlockVector<double> &src_ref);

  template <class DOFINFO>
  void initialize_info(DOFINFO &info, bool face) const;

  template <class DOFINFO>
  void assemble(const DOFINFO &info);

  template <class DOFINFO>
  void assemble(const DOFINFO &info1, const DOFINFO &info2);

 protected:
  std::vector<int> octant_to_global;
  dealii::BlockVector<double> *dst;
  dealii::BlockVector<double> *src;
};

#endif  // AETHER_SN_SWEEPER_H_