module;

#include "defs.hpp"
#include <ksn/ksn.hpp>

#include <vector>



export module diploma.lin_alg;
import diploma.utility;


EXPORT_BEGIN

struct tensor_dims
{
	size_t height{};
	size_t width{};
	size_t depth{};

	size_t total() const { return safe_mul(depth, this->image_size()); }
	size_t image_size() const { return safe_mul(width, height); }

	bool operator==(const tensor_dims&) const = default;

	size_t to_linear_idx(size_t y, size_t x, size_t z) const
	{
		size_t idx = 0;
		idx = safe_fma(idx, depth, z);
		idx = safe_fma(idx, height, y);
		idx = safe_fma(idx, width, x);
		return idx;
	}

	//size_t dimentionality() const noexcept
	//{
	//	size_t d = 3;
	//	if (depth > 1) return d; else --d;
	//	if (width > 1) return d; else --d;
	//	if (height > 1) return d; else --d;
	//	return d;
	//}
};

class tensor
{
	using idx_t = std::ptrdiff_t;

	struct M
	{
		std::vector<fp> data;
		tensor_dims dims;
	} m;

	template<class T>
	using nl = std::numeric_limits<T>;

	static constexpr idx_t _default_idx = nl<idx_t>::is_signed ? nl<idx_t>::min() + 1 : nl<idx_t>::max();
	static constexpr idx_t default_idx = DO_DEBUG_CHECKS ? _default_idx : 0;

public:
	tensor() = default;
	tensor(tensor_dims dims)
	{
		this->create_storage(dims);
	}
	void create_storage(tensor_dims dims)
	{
		m.data.resize(dims.total());
		m.dims = dims;
	}
	size_t total() const noexcept { return m.dims.total(); };

	auto dims() const noexcept { return m.dims; }

	auto data() noexcept { return m.data.data(); }
	auto data() const noexcept { return m.data.data(); }

	auto begin() { return data(); }
	auto end() { return begin() + total(); }

	fp& operator()(idx_t y = default_idx, idx_t x = default_idx, idx_t z = default_idx)
	{
		return data()[to_linear_idx_with_checks(y, x, z)];
	}
	const fp& operator()(idx_t y = default_idx, idx_t x = default_idx, idx_t z = default_idx) const
	{
		return data()[to_linear_idx_with_checks(y, x, z)];
	}

private:
	idx_t to_linear_idx_with_checks(idx_t y, idx_t x, idx_t z) const
	{
		adjust_default_indexes(y, x, z);
		check_no_overflow(x, dims().width);
		check_no_overflow(y, dims().height);
		check_no_overflow(z, dims().depth);
		return dims().to_linear_idx(y, x, z);
	}

	static void check_no_overflow(idx_t idx, size_t dim)
	{
		if constexpr (DO_DEBUG_CHECKS)
			xassert(idx >= 0 && (size_t)idx < dim, "tensor::operator(): invalid index: dim = {}, idx = {}", dim, idx);
	}
	void adjust_default_indexes(idx_t& y, idx_t& x, idx_t& z) const
	{
		if constexpr (DO_DEBUG_CHECKS)
		{
			if (!try_adjust_default_index(z, m.dims.depth)) return;
			if (!try_adjust_default_index(x, m.dims.width)) return;
			if (!try_adjust_default_index(y, m.dims.height)) return;
		}
	}
	static bool try_adjust_default_index(idx_t& idx, size_t dim)
	{
		if (idx != default_idx)
			return false;
		xassert(dim == 1, "Wrong number of arguments for operator() on tensor");
		idx = 0;
		return true;
	}
};

EXPORT_END
