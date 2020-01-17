#include <deal.II/base/config.h>
#include <deal.II/base/exceptions.h>

#include <deal.II/lac/block_indices.h>
#include <deal.II/lac/block_vector_base.h>
#include <deal.II/lac/vector_operation.h>
#include <deal.II/lac/vector_type_traits.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

template <typename Number>
class BlockBlockVector : 
    public dealii::BlockVectorBase<dealii::BlockVector<Number>> {
 public:
  using BaseClass = BlockVectorBase<Vector<Number>>;

  using BlockType = typename BaseClass::BlockType;

  using value_type      = typename BaseClass::value_type;
  using real_type       = typename BaseClass::real_type;
  using pointer         = typename BaseClass::pointer;
  using const_pointer   = typename BaseClass::const_pointer;
  using reference       = typename BaseClass::reference;
  using const_reference = typename BaseClass::const_reference;
  using size_type       = typename BaseClass::size_type;
  using iterator        = typename BaseClass::iterator;
  using const_iterator  = typename BaseClass::const_iterator;

  explicit BlockVector(const unsigned int n_blocks = 0,
                       const unsigned int n_blocks_per_block = 0
                       const size_type    block_size = 0);

  BlockBlockVector(const BlockBlockVector<Number> &V);

  BlockBlockVector(BlockBlockVector<Number> && /*v*/) noexcept = default;

  template <typename OtherNumber>
  explicit BlockVector(const BlockVector<OtherNumber> &v);

  BlockBlockVector(const std::vector<std::vector<size_type>> &block_sizes);

  ~BlockBlockVector() override = default;

  bool has_ghost_elements() const;

  BlockBlockVector& operator=(const value_type s);

  BlockBlockVector<Number>& operator=(const BlockVector<Number> &v);

  BlockBlockVector<Number>& operator=(BlockBlockVector<Number> && /*v*/) =
      default;

  template <class Number2> 
  BlockBlockVector<Number>& operator=(const BlockBlockVector<Number2> &V);

  BlockBlockVector<Number>& operator=(const Vector<Number> &V);

  void reinit(const unsigned int n_blocks,
              const unsigned int n_blocks_per_block,
              const size_type    block_size           = 0,
              const bool         omit_zeroing_entries = false);

  void reinit(const std::vector<std::vector<size_type>> &block_sizes,
              const bool omit_zeroing_entries = false);

  template <typename Number2>
  void
  reinit(const BlockVector<Number2> &V,
         const bool                  omit_zeroing_entries = false);

  template <class BlockVector2>
  void scale(const BlockVector2 &v);

  void swap(BlockVector<Number> &v);

  DeclException0(ExcIteratorRangeDoesNotMatchVectorSize);
};

/*-------------------------Template functions---------------------------------*/

template <typename Number>
BlockBlockVector<Number>::BlockBlockVector(const unsigned int n_blocks,
                                           const unsigned int n_blocks_per_block,
                                           const size_type    block_size) {
  reinit(n_blocks, n_blocks_per_block, block_size);
}

template <typename Number>
BlockBlockVector<Number>::BlockBlockVector(
    const std::vector<size_type> &block_sizes) {
  reinit(block_sizes, false);
}

template <typename Number>
BlockBlockVector<Number>::BlockBlockVector(const BlockVector<Number> &v)
  : BlockVectorBase<Vector<Number>>() {
  this->components.resize(v.n_blocks());
  this->block_indices = v.block_indices;

  for (size_type i = 0; i < this->n_blocks(); ++i)
    this->components[i] = v.components[i];
}

template <typename Number>
template <typename OtherNumber>
BlockVector<Number>::BlockVector(const BlockVector<OtherNumber> &v)
{
  reinit(v, true);
  *this = v;
}

template <typename Number>
void
BlockVector<Number>::reinit(const unsigned int n_blocks,
                            const unsigned int n_blocks_per_block,
                            const size_type    block_size,
                            const bool         omit_zeroing_entries)
{
  std::vector<std::vector<size_type>> block_sizes(
      n_blocks, std::vector<size_type>(n_blocks_per_block, block_size));
  reinit(block_sizes, omit_zeroing_entries);
}

template <typename Number>
void BlockVector<Number>::reinit(
    const std::vector<std::vector<size_type>> &block_sizes,
    const bool omit_zeroing_entries)
{
  this->block_indices.reinit(block_sizes);
  if (this->components.size() != this->n_blocks())
    this->components.resize(this->n_blocks());

  for (size_type i = 0; i < this->n_blocks(); ++i)
    this->components[i].reinit(block_sizes[i], omit_zeroing_entries);
}

template <typename Number>
void
BlockVector<Number>::reinit(const BlockIndices &n,
                            const bool          omit_zeroing_entries)
{
  this->block_indices = n;
  if (this->components.size() != this->n_blocks())
    this->components.resize(this->n_blocks());

  for (size_type i = 0; i < this->n_blocks(); ++i)
    this->components[i].reinit(n.block_size(i), omit_zeroing_entries);
}

template <typename Number>
template <typename Number2>
void
BlockVector<Number>::reinit(const BlockVector<Number2> &v,
                            const bool                  omit_zeroing_entries)
{
  this->block_indices = v.get_block_indices();
  if (this->components.size() != this->n_blocks())
    this->components.resize(this->n_blocks());

  for (size_type i = 0; i < this->n_blocks(); ++i)
    this->block(i).reinit(v.block(i), omit_zeroing_entries);
}



#ifdef DEAL_II_WITH_TRILINOS
template <typename Number>
inline BlockVector<Number> &
BlockVector<Number>::operator=(const TrilinosWrappers::MPI::BlockVector &v)
{
  BaseClass::operator=(v);
  return *this;
}
#endif


template <typename Number>
void
BlockVector<Number>::swap(BlockVector<Number> &v)
{
  std::swap(this->components, v.components);

  dealii::swap(this->block_indices, v.block_indices);
}



template <typename Number>
void
BlockVector<Number>::print(std::ostream &     out,
                           const unsigned int precision,
                           const bool         scientific,
                           const bool         across) const
{
  for (size_type i = 0; i < this->n_blocks(); ++i)
    {
      if (across)
        out << 'C' << i << ':';
      else
        out << "Component " << i << std::endl;
      this->components[i].print(out, precision, scientific, across);
    }
}



template <typename Number>
void
BlockVector<Number>::block_write(std::ostream &out) const
{
  for (size_type i = 0; i < this->n_blocks(); ++i)
    this->components[i].block_write(out);
}



template <typename Number>
void
BlockVector<Number>::block_read(std::istream &in)
{
  for (size_type i = 0; i < this->n_blocks(); ++i)
    this->components[i].block_read(in);
}

/*----------------------- Inline functions ----------------------------------*/



template <typename Number>
template <typename InputIterator>
BlockBlockVector<Number>::BlockBlockVector(const std::vector<size_type> &block_sizes,
                                const InputIterator           first,
                                const InputIterator           end)
{
  // first set sizes of blocks, but
  // don't initialize them as we will
  // copy elements soon
  (void)end;
  reinit(block_sizes, true);
  InputIterator start = first;
  for (size_type b = 0; b < block_sizes.size(); ++b)
    {
      InputIterator end = start;
      std::advance(end, static_cast<signed int>(block_sizes[b]));
      std::copy(start, end, this->block(b).begin());
      start = end;
    };
  Assert(start == end, ExcIteratorRangeDoesNotMatchVectorSize());
}



template <typename Number>
inline BlockVector<Number> &
BlockVector<Number>::operator=(const value_type s)
{
  AssertIsFinite(s);

  BaseClass::operator=(s);
  return *this;
}



template <typename Number>
inline BlockVector<Number> &
BlockVector<Number>::operator=(const BlockVector<Number> &v)
{
  reinit(v, true);
  BaseClass::operator=(v);
  return *this;
}



template <typename Number>
inline BlockVector<Number> &
BlockVector<Number>::operator=(const Vector<Number> &v)
{
  BaseClass::operator=(v);
  return *this;
}



template <typename Number>
template <typename Number2>
inline BlockVector<Number> &
BlockVector<Number>::operator=(const BlockVector<Number2> &v)
{
  reinit(v, true);
  BaseClass::operator=(v);
  return *this;
}

template <typename Number>
inline void
BlockVector<Number>::compress(::VectorOperation::values operation)
{
  for (size_type i = 0; i < this->n_blocks(); ++i)
    this->components[i].compress(operation);
}



template <typename Number>
inline bool
BlockVector<Number>::has_ghost_elements() const
{
  return false;
}



template <typename Number>
template <class BlockVector2>
void
BlockVector<Number>::scale(const BlockVector2 &v)
{
  BaseClass::scale(v);
}

#endif // DOXYGEN


template <typename Number>
inline void
swap(BlockVector<Number> &u, BlockVector<Number> &v)
{
  u.swap(v);
}
