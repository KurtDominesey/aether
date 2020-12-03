#ifndef AETHER_BASE_BLOCK_BLOCK_VECTOR_H_
#define AETHER_BASE_BLOCK_BLOCK_VECTOR_H_

#include <deal.II/lac/block_vector.h>

namespace aether {

/**
 * An implementation of nested block vectors based on deal.II vectors.
 */
template <typename Number>
class BlockBlockVector : 
    public dealii::BlockVectorBase<dealii::BlockVector<Number>> {
 public:
  /**
   * Alias the base class for simpler access.
   */
  using BaseClass =
      typename dealii::BlockVectorBase<dealii::BlockVector<Number>>;

  /**
   * Alias the type of the underlying block vector.
   */
  using BlockType = typename BaseClass::BlockType;

  /**
   * Import the aliases from the base class.
   */
  using value_type      = typename BaseClass::value_type;
  using real_type       = typename BaseClass::real_type;
  using pointer         = typename BaseClass::pointer;
  using const_pointer   = typename BaseClass::const_pointer;
  using reference       = typename BaseClass::reference;
  using const_reference = typename BaseClass::const_reference;
  using size_type       = typename BaseClass::size_type;
  using iterator        = typename BaseClass::iterator;
  using const_iterator  = typename BaseClass::const_iterator;

  /**
   * Constructor. There are four ways to use this constructor. First, without
   * any arguments, it generates an object with no blocks. Given one argument
   * it initializes <tt>n_blocks<\tt> blocks, but these blocks have no
   * sub-blocks. With two arguments, it intializes each block with 
   * <tt>n_subblocks</tt> sub-blocks of size zero. The fourth variant finally
   * intializes all sub-blocks to the same size <tt>block_size</tt>.
   * 
   * Confer the constructor further down if you intend to use blocks of
   * different sizes.
   */
  explicit BlockBlockVector(const unsigned int n_blocks = 0,
                            const unsigned int n_subblocks = 0,
                            const size_type block_size = 0);
  
  /**
   * Copy constructor.
   */
  BlockBlockVector(const BlockBlockVector<Number> &other);

  /**
   * Move constructor.
   */
  BlockBlockVector(BlockBlockVector<Number> &&other) noexcept = default;

  /**
   * Constructor. Set the number of blocks to the size of the outer vector,
   * the number of sub-blocks to be the size of the inner vectors, and
   * initialize each block with <tt>block_sizes[i][j]</tt> zero elements.
   */
  BlockBlockVector(const std::vector<std::vector<size_type>> &block_sizes);

  /**
   * Destructor. Clears memory.
   */
  ~BlockBlockVector() override = default;

  /**
   * Copy operator. Fill all components of the vector with the given scalar
   * value.
   */
  BlockBlockVector& operator=(const value_type value);

  /**
   * Copy operator for arguments of the same type. Resize the present vector
   * if necessary.
   */
  BlockBlockVector<Number>& operator=(const BlockBlockVector<Number> &other);

  /**
   * Move the given vector. This operator replaces the present vector with the
   * contents of the given argument vector.
   */
  BlockBlockVector<Number>& operator=(
      BlockBlockVector<Number> &&other) = default;
  
  /**
   * See matching constructor.
   * 
   * If <tt>omit_zeroing entries==false</tt>, the vector is filled with zeros.
   */
  void reinit(const unsigned int n_blocks,
              const unsigned int n_subblocks = 0,
              const size_type block_size = 0,
              const bool omit_zeroing_entries = false);
  
  /**
   * See matching constructor.
   * 
   * If <tt>omit_zeroing entries==false</tt>, the vector is filled with zeros.
   */
  void reinit(const std::vector<std::vector<size_type>> &block_sizes,
              const bool omit_zeroing_entries = false);

  /**
   * See matching constructor.
   * 
   * If <tt>omit_zeroing entries==false</tt>, the vector is filled with zeros.
   */
  void reinit(const BlockBlockVector<Number> &other, 
              const bool omit_zeroing_entries=false);
};

// inline functions

template <typename Number>
inline BlockBlockVector<Number>& BlockBlockVector<Number>::operator=(
    const value_type value) {
  BaseClass::operator=(value);
  return *this;
}

template <typename Number>
inline BlockBlockVector<Number>& BlockBlockVector<Number>::operator=(
    const BlockBlockVector<Number> &other) {
  reinit(other, true);
  BaseClass::operator=(other);
  return *this;
}

}

#endif  // AETHER_BASE_BLOCK_BLOCK_VECTOR_H_