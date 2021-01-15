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
void Transport<dim, qdim>::stream_add(
    dealii::Vector<double> &dst, const dealii::Vector<double> &src,
    const std::vector<dealii::BlockVector<double>> &boundary_conditions) const {
  dealii::BlockVector<double> dst_b(this->quadrature.size(),
                                    this->dof_handler.n_dofs());
  dealii::BlockVector<double> src_b(this->quadrature.size(),
                                    this->dof_handler.n_dofs());
  dst_b = dst;
  src_b = src;
  stream_add(dst_b, src_b, boundary_conditions);
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
void Transport<dim, qdim>::collide_add(
    dealii::Vector<double> &dst, const dealii::Vector<double> &src) const {
  dealii::BlockVector<double> dst_b(this->quadrature.size(), 
                                    this->dof_handler.n_dofs());
  dealii::BlockVector<double> src_b(this->quadrature.size(), 
                                    this->dof_handler.n_dofs());
  dst_b = dst;
  src_b = src;
  collide_add(dst_b, src_b);
  dst = dst_b;
}

template <int dim, int qdim>
void Transport<dim, qdim>::collide(
      dealii::Vector<double> &dst,
      const dealii::Vector<double> &src,
      const std::vector<double> &cross_sections) const {
  dealii::BlockVector<double> dst_b(this->quadrature.size(), 
                                    this->dof_handler.n_dofs());
  dealii::BlockVector<double> src_b(this->quadrature.size(), 
                                    this->dof_handler.n_dofs());
  src_b = src;
  collide(dst_b, src_b, cross_sections);
  dst = dst_b;
}

template <int dim, int qdim>
void Transport<dim, qdim>::collide_add(
    dealii::Vector<double> &dst, 
    const dealii::Vector<double> &src,
    const std::vector<double> &cross_sections) const {
  dealii::BlockVector<double> dst_b(this->quadrature.size(), 
                                    this->dof_handler.n_dofs());
  dealii::BlockVector<double> src_b(this->quadrature.size(), 
                                    this->dof_handler.n_dofs());
  dst_b = dst;
  src_b = src;
  collide_add(dst_b, src_b, cross_sections);
  dst = dst_b;
}

template <int dim, int qdim>
void Transport<dim, qdim>::stream(
    dealii::BlockVector<double> &dst,
    const dealii::BlockVector<double> &src,
    const std::vector<dealii::BlockVector<double>> &boundary_conditions) const {
  dst = 0;
  stream_add(dst, src, boundary_conditions);
}

template <int dim, int qdim>
void Transport<dim, qdim>::stream_add(
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
            int n_refl = 
                this->quadrature.reflected_index(n, matrices.normals[f][0]);
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
  dst = 0;
  collide_add(dst, src);
}

template <int dim, int qdim>
void Transport<dim, qdim>::collide_add(dealii::BlockVector<double> &dst,
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
        for (int j = 0; j < mass.m(); ++j) {
          dst.block(n)[dof_indices[i]] += mass[i][j] * 
                                          src.block(n)[dof_indices[j]];
        }
      }
    }
  }
}

template <int dim, int qdim>
void Transport<dim, qdim>::collide(dealii::BlockVector<double> &dst,
                                   const dealii::BlockVector<double> &src,
                                   const std::vector<double> &cross_sections) 
                                   const {
  dst = 0;
  collide_add(dst, src, cross_sections);
}

template <int dim, int qdim>
void Transport<dim, qdim>::collide_add(dealii::BlockVector<double> &dst,
                                       const dealii::BlockVector<double> &src,
                                       const std::vector<double> &cross_sections) 
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
    const double cross_section = cross_sections[cell->material_id()];
    cell->get_dof_indices(dof_indices);
    for (int n = 0; n < this->quadrature.size(); ++n) {
      for (int i = 0; i < mass.n(); ++i) {
        for (int j = 0; j < mass.m(); ++j) {
          dst.block(n)[dof_indices[i]] += cross_section * mass[i][j] * 
                                          src.block(n)[dof_indices[j]];
        }
      }
    }
  }
}

template <int dim, int qdim>
void Transport<dim, qdim>::collide_ordinate(dealii::Vector<double> &dst,
                                            const dealii::Vector<double> &src) 
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
    for (int i = 0; i < mass.n(); ++i) {
      dst[dof_indices[i]] = 0;
      for (int j = 0; j < mass.m(); ++j) {
        dst[dof_indices[i]] += mass[i][j] * src[dof_indices[j]];
      }
    }
  }
}

template <int dim, int qdim>
double Transport<dim, qdim>::inner_product(
    const dealii::Vector<double> &left, 
    const dealii::Vector<double> &right) const {
  dealii::BlockVector<double> left_b(this->quadrature.size(),
                                     this->dof_handler.n_dofs());
  dealii::BlockVector<double> right_b(this->quadrature.size(),
                                      this->dof_handler.n_dofs());
  left_b = left;
  right_b = right;
  return inner_product(left_b, right_b);
}

template <int dim, int qdim>
double Transport<dim, qdim>::inner_product(
    const dealii::BlockVector<double> &left, 
    const dealii::BlockVector<double> &right) const {
  dealii::BlockVector<double> right_l2(right);
  collide(right_l2, right);
  double sq = 0;
  for (int n = 0; n < this->quadrature.size(); ++n)
    sq += this->quadrature.weight(n) * (left.block(n) * right_l2.block(n));
  return std::sqrt(sq);
}

template class Transport<1>;
template class Transport<2>;
template class Transport<3>;

}  // namespace aether::pgd::sn