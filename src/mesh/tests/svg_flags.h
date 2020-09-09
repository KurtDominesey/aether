#ifndef AETHER_MESH_TESTS_SVG_FLAGS_H_
#define AETHER_MESH_TESTS_SVG_FLAGS_H_

#include <deal.II/grid/grid_out.h>

namespace aether {

inline dealii::GridOutFlags::Svg svg_flags() {
  dealii::GridOutFlags::Svg svg;
  svg.coloring = dealii::GridOutFlags::Svg::Coloring::material_id;
  svg.margin = false;
  svg.label_cell_index = false;
  svg.label_level_number = false;
  svg.label_level_subdomain_id = false;
  svg.label_material_id = false;
  svg.label_subdomain_id = false;
  svg.draw_colorbar = false;
  svg.draw_legend = false;
  return svg;
}

}

#endif  // AETHER_MESH_TESTS_SVG_FLAGS_H_