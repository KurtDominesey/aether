#ifndef AETHER_SN_FIXED_SOURCE_H_
#define AETHER_SN_FIXED_SOURCE_H_

#include "within_group.h"
#include "scattering_block.h"
#include "discrete_to_moment.h"
#include "moment_to_discrete.h"

namespace aether::pgd::sn {
template <int dim, int qdim> class FixedSourceP;
template <int dim, int qdim> class FixedSourceS;
template <int dim, int qdim> class FixedSourceSGS;
}

namespace aether::sn {

template <class SolverType, int dim, int qdim> class FixedSourceGS;

/**
 * Fixed-source operator.
 * 
 * Implements \f$
 * A_{FS}=\begin{bmatrix}
 * A_{WG,1} & \ldots & S_{g'\rightarrow 1} & \ldots & S_{G\rightarrow 1} \\
 * \vdots & \ddots & & & \vdots \\
 * S_{1\rightarrow g} & & A_{WG,g} & & S_{G\rightarrow g} \\
 * \vdots & & & \ddots & \vdots \\
 * S_{G\rightarrow G} & \ldots & S_{g'\rightarrow G} & \ldots & A_{WG,G} \\
 * \end{bmatrix}\f$ where 
 * \f$S_{g'\rightarrow g}=-BM\Sigma_{s,g'\rightarrow g}D\f$.
 * 
 * The diagonal blocks are stored in @ref FixedSource.within_groups, the upper
 * triangular blocks in @ref FixedSource.upscattering, and the lower triangular
 * blocks in @ref FixedSource.downscattering.
 */
template <int dim, int qdim = dim == 1 ? 1 : 2>
class FixedSource {
 public:
  /**
   * Constructor.
   */
  FixedSource(std::vector<WithinGroup<dim, qdim>> &within_groups,
              std::vector<std::vector<ScatteringBlock<dim>>> &downscattering,
              std::vector<std::vector<ScatteringBlock<dim>>> &upscattering,
              MomentToDiscrete<dim, qdim> &m2d,
              DiscreteToMoment<dim, qdim> &d2m);
  /**
   * Matrix-vector multiplication by fixed-source operator.
   */
  void vmult(dealii::BlockVector<double> &dst,
             const dealii::BlockVector<double> &src,
             bool transposing=false) const;
  /**
   * Transpose matrix-vector multiplication by fixed-source operator.
   */
  template <typename VectorType>
  void Tvmult(VectorType &dst, const VectorType &src) const;
  /**
   * Number of rows.
   */
  int m() const;
  /**
   * Number of columns.
   */
  int n() const;
  //! Diagonal within-group blocks, \f$A_{WG,g}\f$
  const std::vector<WithinGroup<dim, qdim>> &within_groups;
  //! Moment to discrete operator, \f$M\f$
  const MomentToDiscrete<dim, qdim> &m2d;
  //! Whether the matrix is transposed.
  bool transposed = false;

 protected:
  //! Ragged vector of vectors of downscattering blocks, 
  //! \f$\Sigma_{s,g'\rightarrow g}\f$ where \f$g'<g\f$
  const std::vector<std::vector<ScatteringBlock<dim>>> &downscattering;
  //! Ragged vector of vectors of upscattering blocks,
  //! \f$\Sigma_{s,g'\rightarrow g}\f$ where \f$g'>g\f$
  const std::vector<std::vector<ScatteringBlock<dim>>> &upscattering;
  //! Discrete to moment operator, \f$D\f$
  const DiscreteToMoment<dim, qdim> &d2m;
  friend class aether::pgd::sn::FixedSourceP<dim, qdim>;
  friend class aether::pgd::sn::FixedSourceS<dim, qdim>;
  friend class aether::pgd::sn::FixedSourceSGS<dim, qdim>;
  template <class SolverType, int dimm, int qdimm>
  friend class FixedSourceGS;
};

template <int dim, int qdim>
template <typename VectorType>
void FixedSource<dim, qdim>::Tvmult(VectorType &dst, 
                                    const VectorType &src) const {
  vmult(dst, src, true);
}

}  // namespace aether::sn

#endif  // AETHER_SN_FIXED_SOURCE_H_