#include "fixed_source_problem.h"

namespace aether::sn {

template <int dim, int qdim, class TransportType, class TransportBlockType>
void FixedSourceProblem<dim, qdim, TransportType, TransportBlockType>::
    sweep_source(dealii::BlockVector<double> &dst, 
                 const dealii::BlockVector<double> &src) const {
  for (int g = 0; g < within_groups.size(); ++g)
    within_groups[g].transport.vmult(dst.block(g), src.block(g), false);
}

template class FixedSourceProblem<1>;
template class FixedSourceProblem<2>;
template class FixedSourceProblem<3>;

}  // namespace aether::sn