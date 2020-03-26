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

project(c5g7-uo2)

find_package(GTest REQUIRED 1.8.0)
include_directories(${GTEST_INCLUDE_DIRS})

include_directories("../../src")

# set(main "main.cc")
file(GLOB_RECURSE source "[!main]*.cc")
file(GLOB_RECURSE source_lib "../../src/*.cc")
include_directories(${GTEST_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/src)

add_library(lib ${source_lib})
# target_include_directories(lib PUBLIC "../../src")
deal_ii_setup_target(lib)

# set(exec "aether.exe")
set(test "test.exe")
# add_executable(${exec} ${main})
add_executable(${test} ${source})
target_link_libraries(${test} lib)

target_compile_options(${test} PUBLIC ${GTEST_CFLAGS})
target_link_libraries(${test} ${GTEST_BOTH_LIBRARIES})

target_include_directories(${test} PUBLIC "../../src")

DEAL_II_SETUP_TARGET(${test})
# DEAL_II_SETUP_TARGET(${exec})