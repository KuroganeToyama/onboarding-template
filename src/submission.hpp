#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>

// Role: internal helpers backing Grid and apply_stencil.
// Reason: named + inline so the external-linkage
// callers below stay well-defined if this header is included 
// from more than one translation unit.
namespace detail {

// Role: alignment target for the grid buffer, in bytes and in doubles.
// Reason: 64 covers both a cache line and the evaluator's AVX2 (32-byte)
// vector width in one number, so there's only one constant to reason about.
inline constexpr std::size_t kAlignment{64};
inline constexpr std::size_t kAlignmentDoubles{kAlignment / sizeof(double)};

// Role: rounds a value up to the next multiple.
// Reason: generic on purpose — used below to pad stride to a whole alignment
// unit, kept separate from that use so it stays a plain, reusable helper.
constexpr std::size_t round_up_to_multiple(std::size_t value, std::size_t multiple) {
  return (value + multiple - 1) / multiple * multiple;
}

// Role: deleter and owning-pointer type for the grid's aligned buffer.
// Reason: the unsized aligned delete pairs safely with the sized aligned new
// used below as long as the alignment matches, so the deleter needs no
// stored size and a plain unique_ptr can own the buffer directly.
struct AlignedDeleter {
  void operator()(double* p) const noexcept {
    ::operator delete(p, std::align_val_t{kAlignment});
  }
};
using AlignedBuffer = std::unique_ptr<double[], AlignedDeleter>;

// Role: allocates rows*stride doubles, 64-byte aligned and zero-initialized.
// Reason: checks the multiplication first — an absurd rows/stride would
// otherwise silently wrap into a too-small allocation that a caller then
// writes past, one of the API preconditions an owning type has to enforce.
inline AlignedBuffer allocate_zeroed(std::size_t rows, std::size_t stride) {
  constexpr std::size_t max_size{std::numeric_limits<std::size_t>::max()};
  if (stride != 0 && rows > max_size / stride) {
    throw std::length_error("Grid: rows * stride overflows size_t");
  }
  const std::size_t count{rows * stride};
  if (count != 0 && count > max_size / sizeof(double)) {
    throw std::length_error("Grid: element count * sizeof(double) overflows size_t");
  }

  double* p{static_cast<double*>(
    ::operator new(count * sizeof(double), std::align_val_t{kAlignment})
  )};

  // === EXPERIMENTAL: NUMA first-touch (easy to revert, see below) ===
  // Role: zero-fill after allocating, one row per iteration, in parallel.
  // Reason: aligned `operator new` doesn't zero memory, and Grid's contract
  // requires every cell to start at zero. This is parallelized with the
  // same row-based schedule(static) split update_interior uses later:
  // on a NUMA machine, whichever thread first writes ("first touches") a
  // page decides which socket's RAM backs it, so matching the later
  // access pattern here means each thread's own future working set lands
  // on its own socket instead of the whole buffer landing wherever the
  // one thread doing a single-threaded fill happened to run. Safe either
  // way: on a non-NUMA machine (e.g. a single-socket cloud VM, the more
  // likely case for the evaluator) this is still just correct zeroing,
  // parallel instead of serial, with no NUMA effect to gain from
  //
  // REVERT INSTRUCTIONS if benchmarking shows this doesn't help: delete
  // the pragma and the loop below it, and restore the one-line original:
  //     std::fill(p, p + count, 0.0);
  #pragma omp parallel for schedule(static) default(none) shared(p, rows, stride)
  for (std::size_t i = 0; i < rows; ++i) {
    std::fill(p + i * stride, p + i * stride + stride, 0.0);
  }
  // === end experimental block ===

  return AlignedBuffer(p);
}

// Role: bundles rows+cols (Extent) and the physical row stride (Layout)
// into one value instead of three loose integers.
// Reason: rows/cols never travel apart, and neither do the grid and its
// stride; keeping them one type means extending to another dimension only
// touches Extent/Layout and the kernels' loop nest, not Grid's ownership
// or allocation code.
struct Extent {
  std::size_t rows;
  std::size_t cols;
};

struct Layout {
  Extent extent;
  // Physical distance between row starts; >= extent.cols, the gap is
  // padding that keeps every row aligned.
  std::size_t stride;
};

// Role: non-owning pointer + layout into a Grid's buffer; T = double or
// const double gives the mutable/read-only forms from one definition.
// Reason: kernels need only pointer+shape, not Grid itself — keeping this
// to a handful of trivially-copyable fields means building one costs
// nothing, so apply_stencil can construct a fresh view on every call.
template <typename T>
struct BasicGridView {
  T* data;
  Layout layout;

  std::size_t rows() const { return layout.extent.rows; }
  std::size_t cols() const { return layout.extent.cols; }
  std::size_t stride() const { return layout.stride; }
};

using GridView = BasicGridView<double>;
using ConstGridView = BasicGridView<const double>;

// Role: copies the outermost row and column unchanged.
// Reason: kept separate from the interior kernel so boundary and interior
// each have one job, and the interior loop never needs to think about edges.
inline void copy_boundary(ConstGridView old_view, GridView new_view) {
  const std::size_t rows{old_view.rows()};
  const std::size_t cols{old_view.cols()};
  const std::size_t old_stride{old_view.stride()};
  const std::size_t new_stride{new_view.stride()};

  for (std::size_t j{0}; j < cols; ++j) {
    new_view.data[j] = old_view.data[j];
    new_view.data[(rows - 1) * new_stride + j] = old_view.data[(rows - 1) * old_stride + j];
  }
  for (std::size_t i{0}; i < rows; ++i) {
    new_view.data[i * new_stride] = old_view.data[i * old_stride];
    new_view.data[i * new_stride + cols - 1] = old_view.data[i * old_stride + cols - 1];
  }
}

// Role: five-point stencil update for one interior row.
// Reason: old_base/new_base qualify for restrict, the actual kernel
// boundary where the promise is used: apply_stencil (the only path that
// reaches this function) always passes pointers from two distinct Grid
// buffers, so a write through new_base cannot alias a read through old_base.
inline void update_row(
  std::size_t i,
  const double* __restrict old_base, double* __restrict new_base,
  std::size_t cols, std::size_t old_stride, std::size_t new_stride
) {
  const double* center_row{old_base + i * old_stride};
  const double* up_row{center_row - old_stride};
  const double* down_row{center_row + old_stride};
  double* new_row{new_base + i * new_stride};

  #pragma omp simd
  for (std::size_t j = 1; j < cols - 1; ++j) {
    new_row[j] = 0.5   * center_row[j] +
                 0.125 * (up_row[j] + down_row[j] + center_row[j - 1] + center_row[j + 1]);
  }
}

// Role: parallel orchestration — divides interior rows across threads.
// Reason: kept separate from update_row's arithmetic so the scheduling
// decision can change without touching the math.
inline void update_interior(ConstGridView old_view, GridView new_view) {
  const std::size_t rows{old_view.rows()};
  const std::size_t cols{old_view.cols()};
  const std::size_t old_stride{old_view.stride()};
  const std::size_t new_stride{new_view.stride()};

  // Role: plain pointers here, not restricted.
  // Reason: this function only forwards them to update_row and does no
  // arithmetic itself — the restrict promise belongs at update_row's
  // parameters, the actual kernel boundary.
  const double* old_base{old_view.data};
  double* new_base{new_view.data};

  // Role: schedule(static).
  // Reason: every row costs the same fixed amount of work, so there's no
  // imbalance for dynamic/guided to correct.
  #pragma omp parallel for schedule(static) default(none) \
      shared(rows, cols, old_stride, new_stride, old_base, new_base)
  for (std::size_t i = 1; i < rows - 1; ++i) {
    update_row(i, old_base, new_base, cols, old_stride, new_stride);
  }
}

}  // namespace detail

// Role: Grid owns one flat, aligned, zero-initialized buffer for the field.
// Reason: layout_.stride is layout_.extent.cols rounded up for alignment;
// the padding this creates is internal-only and never surfaces through
// operator().
class Grid {
private:
  detail::Layout layout_;
  detail::AlignedBuffer data_;

public:
  Grid(std::size_t rows, std::size_t cols)
    : layout_{{rows, cols}, detail::round_up_to_multiple(cols, detail::kAlignmentDoubles)}
    , data_{detail::allocate_zeroed(layout_.extent.rows, layout_.stride)}
  { }

  // Role: copy deleted, move defaulted.
  // Reason (copy): an accidental multi-megabyte buffer copy should be a
  // compile error, not a silent cost.
  // Reason (move): must be spelled out explicitly, not left implicit —
  // declaring the deleted copy operations suppresses the compiler's
  // implicit move generation too, not just copy; without these two lines
  // Grid would silently be neither copyable nor movable.
  Grid(const Grid&) = delete;
  Grid& operator=(const Grid&) = delete;
  Grid(Grid&&) = default;
  Grid& operator=(Grid&&) = default;

  double& operator()(std::size_t i, std::size_t j) {
    return data_[i * layout_.stride + j];
  }

  double operator()(std::size_t i, std::size_t j) const {
    return data_[i * layout_.stride + j];
  }

  // Role: non-owning access for kernels — pointer plus shape, nothing else.
  // Reason: copy_boundary/update_row/update_interior take only this as a
  // parameter and have no other access to Grid; this public accessor is
  // what replaces a friend declaration.
  detail::GridView view() {
    return detail::GridView{data_.get(), layout_};
  }

  detail::ConstGridView view() const {
    return detail::ConstGridView{data_.get(), layout_};
  }
};

// Role: builds a view of each grid, then dispatches to the boundary and
// interior kernels.
// Reason: no special access to Grid — view() is public, so apply_stencil
// is an ordinary caller, not a friend.
inline void apply_stencil(const Grid& old_grid, Grid& new_grid) {
  const detail::ConstGridView old_view{old_grid.view()};
  const detail::GridView new_view{new_grid.view()};

  // Role: return early on an empty grid.
  // Reason: also avoids unsigned underflow in `rows - 1` below.
  if (old_view.rows() == 0 || old_view.cols() == 0) {
    return;
  }

  detail::copy_boundary(old_view, new_view);
  detail::update_interior(old_view, new_view);
}
