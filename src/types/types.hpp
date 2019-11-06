#ifndef AETHER_TYPES_TYPES_H_
#define AETHER_TYPES_TYPES_H_

#include <deal.II/base/types.h>

namespace types {

const dealii::types::boundary_id reflecting_boundary_id 
    = static_cast<dealii::types::boundary_id>(-2);

}  // namespace types

#endif  // AETHER_TYPES_TYPES_H_