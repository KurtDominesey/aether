#ifndef AETHER_FUNCTIONS_ATTENUATED_H_
#define AETHER_FUNCTIONS_ATTENUATED_H_

#include <deal.II/base/function.h>

class Attenuated : public dealii::Function<1> {
 public:
  Attenuated(double cos_angle, double cross_section, double incident,
             double x0, double x1)
      : dealii::Function<1>(),
        cos_angle(cos_angle),
        cross_section(cross_section),
        incident(incident),
        x0(x0),
        x1(x1){};
  double value(const dealii::Point<1> &p,
               const unsigned int /*component*/) const {
    double x = cos_angle > 0 ? x0 : x1;
    double path = (p(0) - x) / cos_angle;
    return incident * std::exp(-cross_section * path);
  };
  double cos_angle;
  double cross_section;
  double incident;
  double x0;
  double x1;
};

#endif  // AETHER_FUNCTIONS_ATTENUATED_H_