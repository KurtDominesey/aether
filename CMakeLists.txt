# cmake -S . -B build
# cmake --build build

cmake_minimum_required(VERSION 2.8.12)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(PETSC_PKG REQUIRED IMPORTED_TARGET PETSc)
pkg_check_modules(SLEPC_PKG REQUIRED IMPORTED_TARGET SLEPc)

find_package(deal.II 9.3.0 QUIET
  HINTS bundled/install/dealii #${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()
IF(NOT ${DEAL_II_WITH_PETSC})
  MESSAGE(FATAL_ERROR "No PETSc")
ENDIF()

project(aether)

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

set(main "src/main.cc")
file(GLOB_RECURSE source "src/[!main]*.cc" "src/*/*.cc" )
include_directories(${GTEST_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/src)

set(exec "aether.exe")
set(test "test.exe")
add_executable(${exec} ${main})
add_executable(${test} ${source})

target_compile_options(${test} PUBLIC ${GTEST_CFLAGS})
target_link_libraries(${test} ${GTEST_BOTH_LIBRARIES})

DEAL_II_SETUP_TARGET(${test})
DEAL_II_SETUP_TARGET(${exec})