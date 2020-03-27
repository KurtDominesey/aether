#include "../example_test.h"

using namespace aether;
using namespace aether::sn;

static const int dim_ = 2;
static const int qdim_ = 2;

class C5G7Test : virtual public ExampleTest<dim_, qdim_> {
 protected:
  const double pitch = 0.63;

  void SetUp() override {
    const std::vector<std::string> materials =  {"water", "uo2"};
    mgxs = std::make_shared<Mgxs>(7, materials.size(), 1);
    read_mgxs(*mgxs, "c5g7.h5", "294K", materials);
    quadrature = QPglc<qdim_>(1, 2);
    mesh_quarter_pincell(mesh, {0.54}, pitch, {0, 1});
    set_all_boundaries_reflecting(mesh);
    mesh.refine_global(0);
  }
};