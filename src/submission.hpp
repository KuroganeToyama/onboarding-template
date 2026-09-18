#pragma once

#include <algorithm>
#include <cstddef>
#include <new>

namespace {

// Cache-line / AVX2 alignment target, in bytes and in doubles
constexpr std::size_t kAlignment{64};
constexpr std::size_t kAlignmentDoubles{kAlignment / sizeof(double)};

// Round value up to the next multiple
constexpr std::size_t round_up_to_multiple(std::size_t value, std::size_t multiple) {
  return (value + multiple - 1) / multiple * multiple;
}

// Non-owning pointer + shape into a Grid's buffer
struct ConstGridView {
  const double* data;
  std::size_t rows;
  std::size_t cols;
  std::size_t stride;
};

struct GridView {
  double* data;
  std::size_t rows;
  std::size_t cols;
  std::size_t stride;
};

// Copy boundary rows and columns unchanged
void copy_boundary(ConstGridView old_view, GridView new_view) {
  const std::size_t rows{old_view.rows};
  const std::size_t cols{old_view.cols};

  for (std::size_t j{0}; j < cols; ++j) {
    new_view.data[j] = old_view.data[j];
    new_view.data[(rows - 1) * new_view.stride + j] = old_view.data[(rows - 1) * old_view.stride + j];
  }
  for (std::size_t i{0}; i < rows; ++i) {
    new_view.data[i * new_view.stride] = old_view.data[i * old_view.stride];
    new_view.data[i * new_view.stride + cols - 1] = old_view.data[i * old_view.stride + cols - 1];
  }
}

// Five-point stencil update for one interior row
void update_row(
  std::size_t i,
  const double* __restrict old_base, double* __restrict new_base,
  std::size_t cols, std::size_t old_stride, std::size_t new_stride
) {
  const double* center_row{old_base + i * old_stride};
  const double* up_row{center_row - old_stride};
  const double* down_row{center_row + old_stride};
  double* new_row{new_base + i * new_stride};

  // OpenMP canonical loop form required for #pragma omp simd
  #pragma omp simd
  for (std::size_t j = 1; j < cols - 1; ++j) {
    new_row[j] = 0.5   * center_row[j] +
                 0.125 * (up_row[j] + down_row[j] + center_row[j - 1] + center_row[j + 1]);
  }
}

// Parallel orchestration: divides interior rows across threads
void update_interior(ConstGridView old_view, GridView new_view) {
  const std::size_t rows{old_view.rows};
  const std::size_t cols{old_view.cols};
  const std::size_t old_stride{old_view.stride};
  const std::size_t new_stride{new_view.stride};

  // old_base and new_base come from two distinct Grid buffers, so a write
  // through new_base cannot alias a read through old_base
  const double* __restrict old_base{old_view.data};
  double* __restrict new_base{new_view.data};

  // Rows are independent; static scheduling since every row costs the same
  #pragma omp parallel for schedule(static) default(none) \
      shared(rows, cols, old_stride, new_stride, old_base, new_base)
  for (std::size_t i = 1; i < rows - 1; ++i) {
    update_row(i, old_base, new_base, cols, old_stride, new_stride);
  }
}

}  // namespace

// Grid: owns one flat, aligned, zero-initialized buffer for the field.
// stride_ is cols_ rounded up for alignment; padding is internal only and
// never exposed through operator()
class Grid {
private:
  std::size_t rows_;
  std::size_t cols_;
  std::size_t stride_;
  double* data_;

  std::size_t element_count() const { return rows_ * stride_; }

public:
  Grid(std::size_t rows, std::size_t cols)
    : rows_{rows}
    , cols_{cols}
    , stride_{round_up_to_multiple(cols, kAlignmentDoubles)}
    , data_{static_cast<double*>(
        ::operator new(element_count() * sizeof(double), std::align_val_t{kAlignment})
      )}
  {
    // Aligned new does not zero memory
    std::fill(data_, data_ + element_count(), 0.0);
  }

  ~Grid() {
    ::operator delete(data_, element_count() * sizeof(double), std::align_val_t{kAlignment});
  }

  // Move-only: buffer ownership should never be silently copied
  Grid(const Grid&) = delete;
  Grid& operator=(const Grid&) = delete;

  Grid(Grid&& other) noexcept
    : rows_{other.rows_}
    , cols_{other.cols_}
    , stride_{other.stride_}
    , data_{other.data_}
  {
    other.data_ = nullptr;
    other.rows_ = other.cols_ = other.stride_ = 0;
  }

  Grid& operator=(Grid&& other) noexcept {
    if (this != &other) {
      ::operator delete(data_, element_count() * sizeof(double), std::align_val_t{kAlignment});

      rows_ = other.rows_;
      cols_ = other.cols_;
      stride_ = other.stride_;
      data_ = other.data_;

      other.data_ = nullptr;
      other.rows_ = other.cols_ = other.stride_ = 0;
    }
    return *this;
  }

  double& operator()(std::size_t i, std::size_t j) {
    return data_[i * stride_ + j];
  }

  double operator()(std::size_t i, std::size_t j) const {
    return data_[i * stride_ + j];
  }

  // Grants apply_stencil access to rows_/cols_ without public accessors
  friend void apply_stencil(const Grid& old_grid, Grid& new_grid);
};

// Build views from both grids, then dispatch to boundary and interior update
inline void apply_stencil(const Grid& old_grid, Grid& new_grid) {
  const std::size_t rows{old_grid.rows_};
  const std::size_t cols{old_grid.cols_};

  // Empty grid: nothing to update, and avoids underflow below
  if (rows == 0 || cols == 0) {
    return;
  }

  const ConstGridView old_view{old_grid.data_, old_grid.rows_, old_grid.cols_, old_grid.stride_};
  const GridView new_view{new_grid.data_, new_grid.rows_, new_grid.cols_, new_grid.stride_};

  copy_boundary(old_view, new_view);
  update_interior(old_view, new_view);
}
