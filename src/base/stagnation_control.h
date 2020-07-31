#ifndef AETHER_BASE_STAGNATION_CONTROL_H_
#define AETHER_BASE_STAGNATION_CONTROL_H_

#include <deal.II/lac/solver_control.h>

namespace aether {

class StagnationControl : public dealii::SolverControl {
 public:
  using dealii::SolverControl::SolverControl;
  virtual ~StagnationControl() override = default;
  virtual dealii::SolverControl::State check(const unsigned int step, 
                                             const double check_value) override;
};

}  // namespace aether

#endif  // AETHER_BASE_STAGNATION_CONTROL_H_