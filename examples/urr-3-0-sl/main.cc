#include "problem.h"

int main() {
  Problem problem(100, 64, 7.566853);
  problem.run_fixed_source();
  problem.run_criticality();
  return 0;
}