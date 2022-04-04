#ifndef AETHER_PGD_SN_TRANSPORT_H_
#define AETHER_PGD_SN_TRANSPORT_H_

#include "sn/transport.h"

namespace aether::pgd::sn {

template <int dim, int qdim = dim == 1 ? 1 : 2>
class Transport : public aether::sn::Transport<dim, qdim> {
 public:
  using aether::sn::Transport<dim, qdim>::Transport;  // inherit constructors
  using Ordinate = typename aether::sn::Transport<dim, qdim>::Ordinate;
  void stream(dealii::Vector<double> &dst, 
              const dealii::Vector<double> &src,
              const std::vector<dealii::BlockVector<double>>
                  &boundary_conditions,
              bool transposing=false) const;
  void stream(dealii::BlockVector<double> &dst,
              const dealii::BlockVector<double> &src,
              const std::vector<dealii::BlockVector<double>>
                  &boundary_conditions,
              bool transposing=false) const;
  void stream_add(dealii::Vector<double> &dst, 
                  const dealii::Vector<double> &src,
                  const std::vector<dealii::BlockVector<double>>
                      &boundary_conditions,
                  bool transposing=false) const;
  void stream_add(dealii::BlockVector<double> &dst,
                  const dealii::BlockVector<double> &src,
                  const std::vector<dealii::BlockVector<double>>
                      &boundary_conditions,
                  bool transposing=false) const;
  void vmult_mass(dealii::Vector<double> &dst,
                  const dealii::Vector<double> &src) const;
  void vmult_mass(dealii::BlockVector<double> &dst,
                  const dealii::BlockVector<double> &src) const;
  void vmult_mass_add(dealii::Vector<double> &dst,
                      const dealii::Vector<double> &src) const;
  void vmult_mass_add(dealii::BlockVector<double> &dst,
                      const dealii::BlockVector<double> &src) const;
  void vmult_mass_inv(dealii::Vector<double> &dst) const;
  void vmult_mass_inv(dealii::BlockVector<double> &dst) const;
  void collide(dealii::Vector<double> &dst,
               const dealii::Vector<double> &src,
               const std::vector<double> &cross_sections) const;
  void collide(dealii::BlockVector<double> &dst,
               const dealii::BlockVector<double> &src,
               const std::vector<double> &cross_sections) const;
  void collide_add(dealii::Vector<double> &dst,
                   const dealii::Vector<double> &src,
                   const std::vector<double> &cross_sections) const;
  void collide_add(dealii::BlockVector<double> &dst,
                   const dealii::BlockVector<double> &src,
                   const std::vector<double> &cross_sections) const;
  void collide_ordinate(dealii::Vector<double> &dst,
                        const dealii::Vector<double> &src) const;
  double inner_product(const dealii::Vector<double> &left,
                       const dealii::Vector<double> &right) const;
  double inner_product(const dealii::BlockVector<double> &left,
                       const dealii::BlockVector<double> &right) const;
  template <typename VectorType>
  double norm(const VectorType &v) const;
};

template <int dim, int qdim>
template <typename VectorType>
double Transport<dim, qdim>::norm(const VectorType &v) const {
  return std::sqrt(inner_product(v, v));
}

}  // namespace aether::pgd::sn

#endif  // AETHER_PGD_SN_TRANSPORT_H_