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

template <int dim>
class CosineFunction : public dealii::Function<dim> {
 public:
  CosineFunction(double period) : period(period) {}
  double value(const dealii::Point<dim> &p, const unsigned int) const {
    switch (dim) {
      case 1:
        return std::cos(period * p(0));
      case 2:
        return std::cos(period * p(0)) *
               std::cos(period * p(1));
      case 3:
        return std::cos(period * p(0)) *
               std::cos(period * p(1)) *
               std::cos(period * p(2));
      default:
        Assert(false, dealii::ExcNotImplemented());
      }
    return 0.0;
  };
  dealii::Tensor<1, dim> gradient(const dealii::Point<dim> &p, 
                                  const unsigned int) const {
    dealii::Tensor<1, dim> result;
    switch (dim) {
      case 1:
        result[0] = -period * std::sin(period * p(0));
        break;
      case 2:
        result[0] = -period * std::sin(period * p(0)) * std::cos(period * p(1));
        result[1] = -period * std::cos(period * p(0)) * std::sin(period * p(1));
        break;
      case 3:
        result[0] = -period * std::sin(period * p(0)) *
                     std::cos(period * p(1)) *
                     std::cos(period * p(2));
        result[1] = -period * std::cos(period * p(0)) *
                     std::sin(period * p(1)) *
                     std::cos(period * p(2));
        result[2] = -period * std::cos(period * p(0)) *
                     std::cos(period * p(1)) *
                     std::sin(period * p(2));
        break;
      default:
        Assert(false, dealii::ExcNotImplemented());
    }
    return result;
  }
 protected:
  double period;
};

#endif  // AETHER_FUCTIONS_FUCTION_LIB_H_