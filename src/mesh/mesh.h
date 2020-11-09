#ifndef AETHER_MESH_H_
#define AETHER_MESH_H_

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_reordering.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_generator.h>

#include "types/types.h"

namespace aether {

void mesh_quarter_pincell(dealii::Triangulation<2> &tria,
                          const std::vector<double> radii,
                          const double pitch,
                          const std::vector<int> materials,
                          const int trans_mani_id = 2,
                          const int sph_mani_id = 1);

void mesh_eighth_pincell(dealii::Triangulation<2> &tria,
                          std::vector<double> radii,
                          const double pitch,
                          std::vector<int> materials,
                          const int trans_mani_id = 2,
                          const int sph_mani_id = 1,
                          bool octagonal = false);

void mesh_eighth_pincell_ul(dealii::Triangulation<2> &tria,
                            std::vector<double> radii,
                            const double pitch,
                            std::vector<int> materials,
                            const int trans_mani_id = 2,
                            const int sph_mani_id = 1,
                            bool octagonal = false);

void mesh_pincell(dealii::Triangulation<2> &tria,
                  const std::vector<double> &radii,
                  const double &pitch,
                  const std::vector<int> &materials);

void mesh_symmetric_quarter_pincell(dealii::Triangulation<2> &tria,
                                    const std::vector<double> &radii,
                                    const double &pitch,
                                    const std::vector<int> &materials,
                                    const int trans_mani_id = 2,
                                    const int sph_mani_id = 1,
                                    bool octagonal = false);

template <int dim>
void set_all_boundaries_reflecting(dealii::Triangulation<dim>& mesh);

void mesh_mox_assembly(dealii::Triangulation<2> &mesh);

void refine_azimuthal(dealii::Triangulation<2> &mesh, int times=1);

void refine_radial(dealii::Triangulation<2> &mesh, int times=1, 
                   const std::vector<int> max_levels={});

}  // namespace aether

#endif  // AETHER_MESH_H_