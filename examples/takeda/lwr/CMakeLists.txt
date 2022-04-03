cmake_minimum_required(VERSION 2.8.12)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(deal.II 9.0.0 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()

project(takeda-lwr)

find_package(GTest REQUIRED 1.8.0)
include_directories(${GTEST_INCLUDE_DIRS})

file(GLOB_RECURSE source_lib "../../../src/*.cc" "../../../src/*/*.cc")

add_library(aether ${source_lib})
target_include_directories(aether PUBLIC "../../../src")
deal_ii_setup_target(aether)

set(test "test.exe")
file(GLOB_RECURSE source "*.cc")
add_executable(${test} "test.cc" ${source})
target_link_libraries(${test} aether)

target_compile_options(${test} PUBLIC ${GTEST_CFLAGS})
target_link_libraries(${test} GTest::GTest)

deal_ii_setup_target(${test})

# set(main "main.cc")
# file(GLOB_RECURSE source "[!main]*.cc")

# set(exec "main.exe")
# add_executable(${exec} ${main} ${source})

# include_directories("../../../src")
# target_include_directories(${exec} PUBLIC "../../../src")

# target_link_libraries(${exec} aether)

# DEAL_II_SETUP_TARGET(${exec})