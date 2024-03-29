##
#  CMake script for the step-4 tutorial program:
##

# Set the name of the project and target:
SET(TARGET "c5g7.exe")
# SET(DDEAL_II_DIR, "/home/kurt/dealii/install")

# Declare all source files the target consists of. Here, this is only
# the one step-X.cc file, but as you expand your project you may wish
# to add other source files as well. If your project becomes much larger,
# you may want to either replace the following statement by something like
# FILE(GLOB_RECURSE TARGET_SRC  "../../src/[!main|!test]*[!test].cpp")
# FILE(GLOB_RECURSE TARGET_INC  "../../src/*.hpp")
# SET(TARGET_SRC ${TARGET_SRC}  ${TARGET_INC}) 
# or switch altogether to the large project CMakeLists.txt file discussed
# in the "CMake in user projects" page accessible from the "User info"
# page of the documentation.
SET(TARGET_SRC
  c5g7.cc
  )

# SET(CMAKE_CXX_STANDARD 17)
# SET(CMAKE_CXX_STANDARD_REQUIRED ON)

# Usually, you will not need to modify anything beyond this point...

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)

FIND_PACKAGE(deal.II 9.0.0 QUIET
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
PROJECT(${TARGET})
DEAL_II_INVOKE_AUTOPILOT()

INCLUDE_DIRECTORIES("../../src")
TARGET_INCLUDE_DIRECTORIES(${TARGET} PUBLIC "../../src")

file(GLOB_RECURSE source_lib "../../src/*.cc" "../../src/*/*.cc")
add_library(aether ${source_lib})
deal_ii_setup_target(aether)
target_link_libraries(${TARGET} aether)