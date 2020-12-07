#ifndef AETHER_PETSC_BLOCK_BLOCK_VECTOR_H_
#define AETHER_PETSC_BLOCK_BLOCK_VECTOR_H_

#include "base/petsc_block_vector.h"

namespace aether::PETScWrappers::MPI {

class BlockBlockVector : public dealii::BlockVectorBase<BlockVector> {
 public:
  /**
   * Alias the base class for simpler access.
   */
  using BaseClass = dealii::BlockVectorBase<BlockVector>;

  /**
   * Alias the type of the underlying vector.
   */
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

  /**
   * Default constructor. Generate an empty vector without any blocks.
   */
  BlockBlockVector() = default;

  /**
   * Constructor. Generate a block vector with @p n_blocks blocks, each with
   * @p n_subblocks, all of which are parallel vectors across @p communicator
   * with @p block_size elements of which @p local_size elements are stored on
   * the present process.
   */
  explicit BlockBlockVector(const unsigned int n_blocks,
                            const unsigned int n_subblocks,
                            const MPI_Comm& communicator,
                            const size_type block_size,
                            const size_type local_size);

  /**
   * Copy constructor. Set all properties of the parallel vector to those of the
   * given argument and copy the elements.
   */
  BlockBlockVector(const BlockBlockVector &other);

  /**
   * Constructor. Set the number of blocks to <tt>block_sizes.size()</tt>, the
   * number of sub-blocks to <tt>block_sizes[i].size()</tt>, and initialize each
   * sub-block with <tt>block_sizes[i][j]</tt> zero elements. The individual
   * sub-blocks are distributed across the given communicator, and each store
   * <tt>local_elements[i][j]</tt> elements on the present process.
   */
  BlockBlockVector(const std::vector<std::vector<size_type>> &block_sizes,
                   const MPI_Comm& communicator,
                   const std::vector<std::vector<size_type>> &local_elements);

  /**
   * Destructor. Clears memory.
   */
  ~BlockBlockVector() override = default;

  /**
   * Copy operator. Fill all components of the vector that are locally stored
   * with the given scalar value.
   */
  BlockBlockVector& operator=(const value_type value);

  /**
   * Copy operator for arguments of the same type.
   */
  BlockBlockVector& operator=(const BlockBlockVector &other);

  /**
   * See matching constructor.
   * 
   * If <tt>omit_zeroing entries==false</tt>, the vector is filled with zeros.
   */
  void reinit(const unsigned int n_blocks,
              const unsigned int n_subblocks,
              const MPI_Comm& communicator,
              const size_type block_size,
              const size_type local_size,
              const bool omit_zeroing_entries = false);

  /**
   * See matching constructor.
   * 
   * If <tt>omit_zeroing entries==false</tt>, the vector is filled with zeros.
   */
  void reinit(const std::vector<std::vector<size_type>> &block_sizes,
              const MPI_Comm& communicator,
              const std::vector<std::vector<size_type>> &local_sizes,
              const bool omit_zeroing_enetries = false);

  /**
   * Change the dimension to that of the vector <tt>other</tt>.
   * 
   * The elements of <tt>other</tt>, i.e. this function is the same as calling
   * <tt>reinit(other.size(), omit_zeroing_entries)</tt>.
   * 
   * Note that you must call this (or the other reinit() functions) function,
   * rather than calling the reinit() functions of an individual block, to allow
   * the block vector to update its caches of vector sizes. If you call reinit()
   * on one of the blocks, then subsequent actions on this object may yield 
   * unpredictable results since they may be routed to the wrong block.
   */
  void reinit(const BlockBlockVector &other, 
              const bool omit_zeroing_entries = false);

  /**
   * Change the number of blocks to <tt>n_blocks</tt>. The individual blocks
   * will get initialized with zero size, so it is assumed that the user resizes
   * the individual blocks by themself in an appropriate way, and calls
   * <tt>collect_sizes</tt> afterwards.
   */
  void reinit(const unsigned int n_blocks);

  /**
   * Return a reference to the MPI communicator object in use with this vector.
   */
  const MPI_Comm& get_mpi_communicator() const;
};

// inline functions

inline BlockBlockVector::BlockBlockVector(const unsigned int n_blocks,
                                          const unsigned int n_subblocks,
                                          const MPI_Comm& communicator,
                                          const size_type block_size,
                                          const size_type local_size) {
  reinit(n_blocks, n_subblocks, communicator, block_size, local_size);
}

inline BlockBlockVector::BlockBlockVector(
    const std::vector<std::vector<size_type>> &block_sizes,
    const MPI_Comm& communicator,
    const std::vector<std::vector<size_type>> &local_elements) {
  reinit(block_sizes, communicator, local_elements, false);
}

inline BlockBlockVector::BlockBlockVector(const BlockBlockVector &other)
    : BaseClass() {
  this->components.resize(other.n_blocks());
  this->block_indices = other.block_indices;
  for (size_type i = 0; i < this->n_blocks(); ++i) {
    this->components[i] = other.components[i];
  }
}

inline BlockBlockVector& BlockBlockVector::operator=(const value_type value) {
  BaseClass::operator=(value);
  return *this;
}

inline BlockBlockVector& BlockBlockVector::operator=(
    const BlockBlockVector &other) {
  Assert(n_blocks() == 0 || n_blocks() == other.n_blocks(),
         dealii::ExcDimensionMismatch(n_blocks(), other.n_blocks()));
  if (this->n_blocks() != other.n_blocks())
    reinit(other.n_blocks());
  for (size_type i = 0; i < this->n_blocks(); ++i) {
    this->components[i] = other.block(i);
  }
  collect_sizes();
  return *this;
}

inline void BlockBlockVector::reinit(const unsigned int n_blocks,
                                     const unsigned int n_subblocks,
                                     const MPI_Comm& communicator,
                                     const size_type block_size,
                                     const size_type local_size,
                                     const bool omit_zeroing_entries) {
  const std::vector<std::vector<size_type>> block_sizes(n_blocks,
      std::vector<size_type>(n_subblocks, block_size));
  const std::vector<std::vector<size_type>> local_sizes(n_blocks,
      std::vector<size_type>(n_subblocks, local_size));
  reinit(block_sizes, communicator, local_sizes, omit_zeroing_entries);
}

inline void BlockBlockVector::reinit(
    const std::vector<std::vector<size_type>> &block_sizes,
    const MPI_Comm& communicator,
    const std::vector<std::vector<size_type>> &local_sizes,
    const bool omit_zeroing_entries) {
  std::vector<size_type> flat_sizes(block_sizes.size());
  for (int i = 0; i < block_sizes.size(); ++i)
    for (int j = 0; j < block_sizes[i].size(); ++j)
      flat_sizes[i] += block_sizes[i][j];
  this->block_indices.reinit(flat_sizes);
  if (this->components.size() != this->n_blocks())
    this->components.resize(this->n_blocks());
  for (size_type i = 0; i < this->n_blocks(); ++i)
    this->components[i].reinit(block_sizes[i], communicator, local_sizes[i], 
                               omit_zeroing_entries);
}

inline void BlockBlockVector::reinit(const BlockBlockVector &other, 
                                     const bool omit_zeroing_entries) {
  this->block_indices = other.get_block_indices();
  if (this->components.size() != this->n_blocks())
    this->components.resize(this->n_blocks());
  for (size_type i = 0; i < this->n_blocks(); ++i)
    block(i).reinit(other.block(i), omit_zeroing_entries);
}

inline const MPI_Comm& BlockBlockVector::get_mpi_communicator() const {
  return block(0).get_mpi_communicator();
}

}  // namespace aether::PETScWrappers::MPI

#endif  // AETHER_PETSC_BLOCK_BLOCK_VECTOR_H_