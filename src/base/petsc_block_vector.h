#ifndef AETHER_BASE_PETSC_BLOCK_VECTOR_H_
#define AETHER_BASE_PETSC_BLOCK_VECTOR_H_

#include <deal.II/lac/petsc_block_vector.h>

namespace aether::PETScWrappers::MPI {

/**
 * A very thin wrapper around deal.II's PETScWrappers::BlockVector with
 * constructors that match those of deal.II's native block vectors.
 * 
 * Allows for easier testing in serial, because the block vector type can be
 * templated.
 */
class BlockVector : virtual public dealii::PETScWrappers::MPI::BlockVector,
                    virtual public dealii::PETScWrappers::VectorBase {
 public:
  using BaseClass = dealii::PETScWrappers::MPI::BlockVector;
  using BlockType = BaseClass::BlockType;

  /**
   * Import the aliases from the base class.
   */
  using value_type      = BaseClass::value_type;
  using pointer         = BaseClass::pointer;
  using const_pointer   = BaseClass::const_pointer;
  using reference       = BaseClass::reference;
  using const_reference = BaseClass::const_reference;
  using size_type       = BaseClass::size_type;
  using iterator        = BaseClass::iterator;
  using const_iterator  = BaseClass::const_iterator;
  
  using real_type       = BlockType::real_type;

  using BaseClass::BlockVector;
  BlockVector(const unsigned int n_blocks, const size_type block_size);
  BlockVector(const std::vector<size_type> &block_sizes);
  BlockVector& operator=(const BaseClass::value_type value);
  BlockVector& operator=(const BlockVector &other);
  using BaseClass::reinit;
  void reinit(const unsigned int n_blocks, const size_type block_size, 
              const bool omit_zeroing_entries = false);
  void reinit(const std::vector<size_type> &block_sizes,
              const bool omit_zeroing_entries = false);
  // Have to override these to prevent ambiguities
  const MPI_Comm& get_mpi_communicator() const override;
  std::size_t size() const;
  void compress(dealii::VectorOperation::values operation);
  value_type operator()(size_type i) const;
  reference operator()(size_type i);
  bool operator==(const dealii::PETScWrappers::VectorBase& other) const;
};

inline BlockVector::BlockVector(const unsigned int n_blocks, 
                                const unsigned int block_size)
    : BaseClass(n_blocks, MPI_COMM_WORLD, block_size, block_size) {}

inline BlockVector::BlockVector(const std::vector<size_type> &block_sizes)
    : BaseClass(block_sizes, MPI_COMM_WORLD, block_sizes) {} 

inline BlockVector& BlockVector::operator=(const BaseClass::value_type value) {
  BaseClass::operator=(value);
  return *this;
}

inline BlockVector& BlockVector::operator=(const BlockVector &other) {
  BaseClass::operator=(other);
  return *this;
}

inline void BlockVector::reinit(const unsigned int n_blocks, 
                                const size_type block_size, 
                                const bool omit_zeroing_entries) {
  BaseClass::reinit(n_blocks, MPI_COMM_WORLD, block_size, block_size, 
                    omit_zeroing_entries);
}

inline void BlockVector::reinit(const std::vector<size_type> &block_sizes, 
                                const bool omit_zeroing_entries) {
  BaseClass::reinit(block_sizes, MPI_COMM_WORLD, block_sizes, 
                    omit_zeroing_entries);
}

inline const MPI_Comm& BlockVector::get_mpi_communicator() const {
  return this->block(0).get_mpi_communicator();
}

inline std::size_t BlockVector::size() const {
  return BaseClass::size();
}

inline void BlockVector::compress(dealii::VectorOperation::values operation) {
  BaseClass::compress(operation);
}

inline BlockVector::value_type BlockVector::operator()(size_type i) const {
  return BaseClass::operator()(i);
}

inline BlockVector::reference BlockVector::operator()(size_type i) {
  return BaseClass::operator()(i);
}

inline bool BlockVector::operator==(
    const dealii::PETScWrappers::VectorBase& other) const {
  auto other_b = dynamic_cast<const BaseClass*>(&other);
  if (other_b != nullptr)
    return BaseClass::operator==(*other_b);
  return dealii::PETScWrappers::VectorBase::operator==(other);
}

}  // namespace aether::PETScWrappers::MPI 

#endif  // AETHER_BASE_PETSC_BLOCK_VECTOR_H_
