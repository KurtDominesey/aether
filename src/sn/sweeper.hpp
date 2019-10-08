#ifndef AETHER_SN_SWEEPER_H_
#define AETHER_SN_SWEEPER_H_

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_vector.h>

class Sweeper {
 public:
  void initialize(std::vector<int> &octant_to_global_,
                  dealii::BlockVector<double> &dst_ref, 
                  const dealii::BlockVector<double> &src_ref,
                  int dofs_per_cell);

  template <class DOFINFO>
  void initialize_info(DOFINFO &info, bool face) const {
    // info.initialize_matrices(octant_to_global.size(), face);
    const int num_ords = octant_to_global.size();
    info.initialize_matrices(num_ords, face);
    info.initialize_vectors(num_ords);
    AssertDimension(num_ords, info.n_matrices());
    AssertDimension(num_ords, info.n_vectors());
  }

  template <class DOFINFO>
  void assemble(const DOFINFO &info) {
    Assert(face_assembled, dealii::ExcInvalidState());
    face_assembled = false;
    std::cout << "assemble cell\n";
    for (int n = 0; n < info.n_matrices(); ++n) {
      int n_global = octant_to_global[n];
      dealii::FullMatrix<double> &matrix = matrices[n];
      matrix.add(1, info.matrix(n).matrix);
      dealii::Vector<double> &vector = vectors[n];
      dealii::Vector<double> local_dst = vector;
      std::cout << "invert\n";
      matrix.print(std::cout);
      vector.print(std::cout);
      matrix.gauss_jordan();  // directly invert
      matrix.vmult(local_dst, vectors[n]);
      dst->block(n_global).add(info.indices, local_dst);
      matrix = 0;
      vector = 0;
    }
  }

  template <class DOFINFO>
  void assemble(const DOFINFO &info1, const DOFINFO &info2) {
    std::cout << "assemble face\n";
    face_assembled = true;
    for (int n = 0; n < info1.n_matrices(); ++n) {
      // info1.matrix(n, true).matrix.print(std::cout);
      // std::cout << std::endl;
      matrices[n].add(1, info1.matrix(n).matrix);
      vectors[n].add(1, info1.vector(n).block(0));
    }
  }

  std::vector<dealii::FullMatrix<double> > matrices;
  std::vector<dealii::Vector<double> > vectors;

 protected:
  bool face_assembled = false;
  std::vector<int> octant_to_global;
  dealii::BlockVector<double> *dst;
  const dealii::BlockVector<double> *src;
};

#endif  // AETHER_SN_SWEEPER_H_