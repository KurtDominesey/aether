#ifndef AETHER_FUNCTIONS_FUNCTION_LIB_H_
#define AETHER_FUNCTIONS_FUNCTION_LIB_H_

#include <deal.II/base/function.h>

template <int dim>
using ScalarFunction = std::function<double(const dealii::Point<dim>&)>;
template <int dim>
using VectorFunction 
    = std::function<dealii::Tensor<1, dim>(const dealii::Point<dim>&)>;

template <int dim>
class Streamed : public dealii::Function<dim> {
 public:
  Streamed(const dealii::Tensor<1, dim> &ordinate, 
           const VectorFunction<dim> &grad)
      : ordinate(ordinate), grad(grad) {}
  double value(const dealii::Point<dim> &p, const unsigned int = 0) const {
    return ordinate * grad(p);
  }
 protected:
  const dealii::Tensor<1, dim> ordinate;
  const VectorFunction<dim> grad;
};

template <int dim>
class Collided : public dealii::Function<dim> {
 public:
  Collided(const double &cross_section, const ScalarFunction<dim> &flux)
      : cross_section(cross_section), flux(flux) {}
  double value(const dealii::Point<dim> &p, const unsigned int = 0) const {
    return cross_section * flux(p);
  }
 protected:
  const double cross_section;
  const ScalarFunction<dim> flux;
};

#endif  // AETHER_FUCTIONS_FUCTION_LIB_H_