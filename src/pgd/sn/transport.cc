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
            int n_refl;
            if (qdim == 2) {
              Assert(this->quadrature.is_tensor_product(), 
                      dealii::ExcNotImplemented());
              const dealii::Quadrature<1> &q_polar = 
                  this->quadrature.get_tensor_basis()[0];
              const dealii::Quadrature<1> &q_azim = 
                  this->quadrature.get_tensor_basis()[1];
              int n_polar = n % q_polar.size();
              int n_azim  = n / q_polar.size();
              Assert(n == n_azim * q_polar.size() + n_polar,
                      dealii::ExcInvalidState());
              dealii::Tensor<1, 2> zero_azim({0, 1});
              dealii::Tensor<1, 2> norm_azim(
                  {matrices.normals[f][0][0], matrices.normals[f][0][1]});
              double theta = std::acos(norm_azim * zero_azim);
              double sector = 2.0 * dealii::numbers::PI
                              / (double)q_azim.size();
              Assert(std::fmod(theta/sector, 1.0) < 1e-12,
                      dealii::ExcInvalidState());
              int n_pivot = std::round(theta/sector);
              int n_wrap = n_pivot - 1 - (n_azim - n_pivot);
              int n_azim_refl = n_wrap >= 0 ? n_wrap : q_azim.size() + n_wrap;
              bool is_normal_z = 
                  dim == 3 
                  && matrices.normals[f][0][0] == 0.0 
                  && matrices.normals[f][0][1] == 0.0 
                  && std::abs(matrices.normals[f][0][2]) == 1.0;
              if (is_normal_z)
                n_azim_refl = n_azim;
              int n_polar_refl = 
                  dim == 2 ? n_polar
                            : (q_polar.size() - 1 - n_polar);
              n_refl = n_azim_refl * q_polar.size() + n_polar_refl;
            } else {
              Assert(qdim == 1, dealii::ExcNotImplemented());
              n_refl = this->quadrature.size() - 1 - n;
            }
            dealii::Tensor<1, dim> ordinate_refl = this->ordinates[n] 
                - 2 * (this->ordinates[n] * matrices.normals[f][0]) 
                * matrices.normals[f][0];
            double error = this->ordinates[n_refl] * ordinate_refl
                            - this->ordinates[n_refl].norm() * ordinate_refl.norm();
            Assert(std::abs(error) < 1e-12,
                    dealii::ExcMessage("Ordinates not reflecting"));
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