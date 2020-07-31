#include "base/stagnation_control.h"

namespace aether {

dealii::SolverControl::State StagnationControl::check(
    const unsigned int step, const double check_value) {
  if (step > 0 && check_value >= lvalue) {
    // if (m_log_result)
    //   dealii::deallog << "Convergence step " << step << " value " << check_value
    //                   << std::endl;
    lstep = step;
    lvalue = check_value;
    lcheck = failure;
    if (history_data_enabled)
      history_data.push_back(check_value);
    return failure;
  }
  return SolverControl::check(step, check_value);
}

}  // namespace aether