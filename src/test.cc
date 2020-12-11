#include <deal.II/base/mpi.h>

#include "gtest/gtest.h"


int main (int argc, char **argv) {
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  ::testing::InitGoogleTest(&argc, argv);
  dealii::deal_II_exceptions::disable_abort_on_exception();
  return RUN_ALL_TESTS();
}