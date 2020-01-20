#include "pgd/sn/transport.h"

namespace aether::pgd::sn {

template <int dim, int qdim>
void Transport<dim, qdim>::stream(
    dealii::Vector<double> &dst, const dealii::Vector<double> &src,
    const std::vector<dealii::BlockVector<double>> &boundary_conditions) const {
  dealii::BlockVector<double> dst_b(this->quadrature.size(),
                                    this->dof_handler.n_dofs());
  dealii::BlockVector<double> src_b(this->quadrature.size(),
                                    this->dof_handler.n_dofs());
  src_b = src;
  stream(dst_b, src_b, boundary_conditions);
  dst = dst_b;
}

template <int dim, int qdim>
void Transport<dim, qdim>::collide(dealii::Vector<double> &dst,
                                   const dealii::Vector<double> &src) const {
  dealii::BlockVector<double> dst_b(this->quadrature.size(), 
                                    this->dof_handler.n_dofs());
  dealii::BlockVector<double> src_b(this->quadrature.size(), 
                                    this->dof_handler.n_dofs());
  src_b = src;
  collide(dst_b, src_b);
  dst = dst_b;
}

template <int dim, int qdim>
void Transport<dim, qdim>::stream(
    dealii::BlockVector<double> &dst,
    const dealii::BlockVector<double> &src,
    const std::vector<dealii::BlockVector<double>> &boundary_conditions) const {
  std::vector<dealii::types::global_dof_index> dof_indices(
      this->dof_handler.get_fe().dofs_per_cell);
  std::vector<dealii::types::global_dof_index> dof_indices_neighbor(
      dof_indices);
  using ActiveCell = typename dealii::DoFHandler<dim>::active_cell_iterator;
  using Cell = typename dealii::DoFHandler<dim>::cell_iterator;
  using Face = typename dealii::DoFHandler<dim>::face_iterator;
  int c = 0;
  for (ActiveCell cell = this->dof_handler.begin_active(); 
       cell != this->dof_handler.end(); ++cell, ++c) {
    if (!cell->is_locally_owned())
      continue;
    const auto &matrices = this->cell_matrices[c];
    cell->get_dof_indices(dof_indices);
    for (int n = 0; n < this->quadrature.size(); ++n) {
      const Ordinate &ordinate = this->ordinates[n];
      for (int i = 0; i < dof_indices.size(); ++i)
        for (int j = 0; j < dof_indices.size(); ++j)
          dst.block(n)[dof_indices[i]] +=
              -(ordinate * matrices.grad[i][j]) * src.block(n)[dof_indices[j]];
      for (int f = 0; f < dealii::GeometryInfo<dim>::faces_per_cell; ++f) {
        if (ordinate * matrices.normals[f][0] > 0) {  // outflow
          for (int i = 0; i < dof_indices.size(); ++i)
            for (int j = 0; j < dof_indices.size(); ++j)
              dst.block(n)[dof_indices[i]] += ordinate *
                                              matrices.outflow[f][i][j] *
                                              src.block(n)[dof_indices[j]];
        } else {  // inflow
          const Face face = cell->face(f);
          if (!face->at_boundary()) {
            // inflow from neighbor
            const Cell neighbor = cell->neighbor(f);
            const int f_neighbor = cell->neighbor_of_neighbor(f);
            Assert(!face->has_children(), dealii::ExcNotImplemented());
            if (!face->has_children()) {
              // one neighbor
              neighbor->get_dof_indices(dof_indices_neighbor);
              for (int i = 0; i < dof_indices.size(); ++i)
                for (int j = 0; j < dof_indices_neighbor.size(); ++j)
                  dst.block(n)[dof_indices[i]] +=
                      ordinate * matrices.inflow[f][0][i][j] *
                      src.block(n)[dof_indices_neighbor[j]];
            } else {
              // multiple neighbors
              for (int f_sub = 0; f_sub < face->number_of_children(); ++f_sub) {
                const Cell &neighbor_child =
                    cell->neighbor_child_on_subface(f, f_sub);
                neighbor_child->get_dof_indices(dof_indices_neighbor);
                for (int i = 0; i < dof_indices.size(); ++i)
                  for (int j = 0; j < dof_indices_neighbor.size(); ++j)
                    dst.block(n)[dof_indices[i]] +=
                        ordinate * matrices.inflow[f][f_sub][i][j] *
                        src.block(n)[dof_indices_neighbor[j]];
              }
            }
          } else if (face->boundary_id() == types::reflecting_boundary_id) {
            // inflow from reflecting boundary
            int n_refl = (n + this->quadrature.size()/2) 
                         % this->quadrature.size();
            for (int i = 0; i < dof_indices.size(); ++i)
              for (int j = 0; j < dof_indices.size(); ++j)
                dst.block(n)[dof_indices[i]] +=
                    ordinate * matrices.outflow[f][i][j] * 
                    src.block(n_refl)[dof_indices[j]];
          } else {
            // inflow from Dirichlet boundary
            for (int i = 0; i < dof_indices.size(); ++i)
              for (int j = 0; j < dof_indices.size(); ++j)
                dst.block(n)[dof_indices[i]] +=
                    ordinate * matrices.outflow[f][i][j] *
                    boundary_conditions[face->boundary_id()].block(n)[j];
          }
        }
      }
    }
  }
}

template <int dim, int qdim>
void Transport<dim, qdim>::collide(dealii::BlockVector<double> &dst,
                                   const dealii::BlockVector<double> &src) 
                                   const {
  std::vector<dealii::types::global_dof_index> dof_indices(
      this->dof_handler.get_fe().dofs_per_cell);
  using ActiveCell = typename dealii::DoFHandler<dim>::active_cell_iterator;
  int c = 0;
  for (ActiveCell cell = this->dof_handler.begin_active();
       cell != this->dof_handler.end(); ++cell, ++c) {
    if (!cell->is_locally_owned())
      continue;
    const dealii::FullMatrix<double> &mass= this->cell_matrices[c].mass;
    cell->get_dof_indices(dof_indices);
    for (int n = 0; n < this->quadrature.size(); ++n) {
      for (int i = 0; i < mass.n(); ++i) {
        dst.block(n)[dof_indices[i]] = 0;
        for (int j = 0; j < mass.m(); ++j) {
          dst.block(n)[dof_indices[i]] += mass[i][j] * 
                                          src.block(n)[dof_indices[j]];
        }
      }
    }
  }
}

template class Transport<1>;
template class Transport<2>;
template class Transport<3>;

}  // namespace aether::pgd::sn