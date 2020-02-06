#ifndef AETHER_MESH_H_
#define AETHER_MESH_H_

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_reordering.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_generator.h>

void mesh_quarter_pincell(dealii::Triangulation<2> &tria,
                          const std::vector<double> radii,
                          const double pitch,
                          const std::vector<int> materials);

void mesh_eighth_pincell(dealii::Triangulation<2> &tria,
                          const std::vector<double> radii,
                          const double pitch,
                          const std::vector<int> materials);

void mesh_pincell(dealii::Triangulation<2> &tria,
                  const std::vector<double> &radii,
                  const double &pitch,
                  const std::vector<int> &materials);

#endif  // AETHER_MESH_H_