#ifndef AETHER_FIXED_SOURCE_GS_H_
#define AETHER_FIXED_SOURCE_GS_H_

#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_richardson.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/linear_operator.h>

#include "sn/within_group.h"
#include "sn/scattering_block.h"
#include "sn/fixed_source.h"

namespace aether::sn {

/**
 * Block Gauss-Seidel for the multigroup fixed source operator.
 */
template <class SolverType, int dim, int qdim = dim == 1 ? 1 : 2>
class FixedSourceGS {
 public:

  /**
   * Constructor.
   * 
   * @param within_groups Within group (diagonal block) operators.
   * @param downscattering Downscattering (lower triangle) operators.
   * @param upscattering Upscattering (upper triangle) operators.
   * @param m2d Moment to discrete operator.
   * @param d2m Discrete to moment operator
   * @param solver Solver to invert the within group operators.
   */
  FixedSourceGS(
      const std::vector<WithinGroup<dim, qdim>> &within_groups,
      const std::vector<std::vector<ScatteringBlock<dim>>> &downscattering,
      const std::vector<std::vector<ScatteringBlock<dim>>> &upscattering,
      const MomentToDiscrete<dim, qdim> &m2d,
      const DiscreteToMoment<dim, qdim> &d2m,
      SolverType &solver);

  FixedSourceGS(const FixedSource<dim, qdim> &fixed_source, SolverType &solver);

  /**
   * Apply the Gauss-Seidel linear operator.
   * 
   * @param dst Destination vector.
   * @param src Source vector.
   */
  void vmult(dealii::BlockVector<double> &dst,
             const dealii::BlockVector<double> &src) const;

  /**
   * Apply the transpose of the Gauss-Seidel linear operator.
   * 
   * @param dst Desination vector.
   * @param src Source vector.
   */
  void Tvmult(dealii::BlockVector<double> &dst,
              const dealii::BlockVector<double> &src) const;

  /**
   * Perform one step of the Gauss-Seidel iteration.
   *
   * @param x Current iterate.
   * @param src Source vector.
   */
  void step(dealii::BlockVector<double> &x,
            const dealii::BlockVector<double> &src) const;

  //! Whether the matrix is transposed
  bool transposed = false;

 protected:
  /**
   * Apply the Gauss-Seidel linear operator (internal method).
   * 
   * @param dst Destination vector.
   * @param src Source vector.
   */
  void do_vmult(dealii::BlockVector<double> &dst,
                const dealii::BlockVector<double> &src) const;

  /**
   * Apply the transpose of the Gauss-Seidel linear operator (internal method).
   *  
   * @param dst Desination vector.
   * @param src Source vector.
   */
  void do_Tvmult(dealii::BlockVector<double> &dst,
                 const dealii::BlockVector<double> &src) const;

  //! Within group (diagonal block) operators.
  const std::vector<WithinGroup<dim, qdim>> &within_groups;
  //! Downscattering (lower triangle) operators.
  const std::vector<std::vector<ScatteringBlock<dim>>> &downscattering;
  //! Upscattering (upper triangle) operators.
  const std::vector<std::vector<ScatteringBlock<dim>>> &upscattering;
  //! Moment to discrete operator.
  const MomentToDiscrete<dim, qdim> &m2d;
  //! Discrete to moment operator.
  const DiscreteToMoment<dim, qdim> &d2m;
  //! Solver to invert the within group operators.
  SolverType &solver;
};

}  // namespace aether::sn

#endif  // AETHER_FIXED_SOURCE_GS_H_ 