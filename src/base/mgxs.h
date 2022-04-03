#ifndef AETHER_BASE_MGXS_H_
#define AETHER_BASE_MGXS_H_

#include <hdf5.h>

#include <deal.II/base/hdf5.h>
#include <deal.II/dofs/dof_handler.h>

#include "sn/transport.h"

namespace aether {

namespace sn {
template <int dim, int qdim> class Transport;
}

/**
 * Multigroup cross-sections.
 */
struct Mgxs {
  /**
   * Constructor.
   */
  Mgxs(int num_groups, int num_materials, int num_legendre)
      : num_groups(num_groups), 
        num_materials(num_materials),
        group_structure(num_groups + 1),
        total(num_groups, std::vector<double>(num_materials)),
        chi(num_groups, std::vector<double>(num_materials)),
        nu_fission(num_groups, std::vector<double>(num_materials)),
        scatter(num_groups, std::vector<std::vector<double>>(
                num_groups, std::vector<double>(num_materials*num_legendre))) {}
  //! Number of groups.
  const int num_groups;
  //! Number of materials.
  const int num_materials;
  //! Deep-copy operator
  Mgxs& operator=(const Mgxs& other);
  //! Group boundaries in eV (electron volts).
  std::vector<double> group_structure;
  //! Total cross-sections by group and material.
  std::vector<std::vector<double>> total;
  //! Fission neutron yield by group and material.
  std::vector<std::vector<double>> chi;
  //! Neutron production from fission by group and material.
  std::vector<std::vector<double>> nu_fission;
  //! Scattering matrix by incoming group, outgoing group, material, and
  //! Legendre order.
  std::vector<std::vector<std::vector<double>>> scatter;
  //! Multiplication by a scalar
  Mgxs& operator*=(const double s);
  //! Division by a scalar
  Mgxs& operator/=(const double s);
};

/**
 * Read MGXS from an OpenMC MGXS HDF5 file.
 */
Mgxs read_mgxs(const std::string &filename, 
               const std::string &temperature,
               const std::vector<std::string> &materials,
               const bool read_structure=false);

/**
 * Read MGXS from an OpenMC MGXS HDF5 file.
 */
void read_mgxs(Mgxs &mgxs,
               const std::string &filename, 
               const std::string &temperature,
               const std::vector<std::string> &materials,
               const bool read_structure=false);

/**
 * Read MGXS from an OpenMC MGXS HDF5 file.
 */
void write_mgxs(const Mgxs& mgxs,
                const std::string &filename, 
                const std::string &temperature,
                const std::vector<std::string> &materials);

/**
 * Spatially integrate Legendre moments of flux in each material zone.
 * 
 * Computes energy_spectra 
 * \f$f_{j,\ell,g}=\int_{\mathcal{V}_j}\phi_{g,\ell}(\vec{r})dr\f$ for every
 * energy group \f$g\f$, Legendre order \f$\ell\f$, and material zone 
 * \f$\mathcal{V}_j\f$ according to the mass matrix of `transport` and `flux`
 * \f$\phi\f$.
 * The flux should be collapsed from either an ordinate-wise or spherical
 * harmonic form by @ref DiscreteToMoment::discrete_to_legendre or
 * @ref DiscreteToMoment::moment_to_legendre respectively before calling this
 * function.
 * 
 * @param spectra Energy spectra, indexed by material, blocked by Legendre order.
 * @param flux Legendre moments of the flux, blocked by energy group.
 * @param dof_handler DoF handler storing the mesh and finite elements.
 * @param transport Transport operator storing the mass matrix.
 * 
 */
template <int dim, int qdim = dim == 1 ? 1 : 2>
void collapse_spectra(std::vector<dealii::BlockVector<double>> &spectra,
                      const dealii::BlockVector<double> &flux,
                      const dealii::DoFHandler<dim> &dof_handler,
                      const sn::Transport<dim, qdim> &transport);

/**
 * Transport correction to apply for @ref collapse_mgxs.
 */
enum TransportCorrection {
  CONSISTENT_P,
  INCONSISTENT_P
  // DIAGONAL,
  // BHS
};

/**
 * Collapse MGXS from reference flux.
 * 
 * Internally calls @ref collapse_spectra then 
 * @ref collapse_mgxs(const std::vector<dealii::BlockVector<double>>&, const Mgxs&, const std::vector<int>&, const TransportCorrection).
 */
template <int dim, int qdim = dim == 1 ? 1 : 2>
Mgxs collapse_mgxs(const dealii::BlockVector<double> &flux,
                   const dealii::DoFHandler<dim> &dof_handler,
                   const sn::Transport<dim, qdim> &transport,
                   const Mgxs &mgxs,
                   const std::vector<int> &g_maxes,
                   const TransportCorrection correction = CONSISTENT_P);

/**
 * Collapse MGXS from reference spectrum.
 */
Mgxs collapse_mgxs(const dealii::Vector<double> &spectrum, 
                   const Mgxs &mgxs, const std::vector<int> &g_maxes,
                   const TransportCorrection correction = CONSISTENT_P);

/**
 * Collapse MGXS from reference spectra.
 */
Mgxs collapse_mgxs(const std::vector<dealii::BlockVector<double>> &spectra,
                   const Mgxs &mgxs, const std::vector<int> &g_maxes,
                   const TransportCorrection correction = CONSISTENT_P);

}

#endif  // AETHER_BASE_MGXS_H_