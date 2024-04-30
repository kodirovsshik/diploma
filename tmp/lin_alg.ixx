
module;

#include "defs.hpp"
#include <ksn/ksn.hpp>

#include <vector>



export module diploma.lin_alg;
import diploma.utility;
import diploma.thread_pool;


EXPORT_BEGIN

struct tensor_dims
{
	size_t height{};
	size_t width{};
	size_t depth{};

	tensor_dims() = default;
	tensor_dims(size_t h, size_t w = 1, size_t d = 1)
		: height(h), width(w), depth(d) {}

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

#define TENSOR_USE_DATA_SPAN _KSN_IS_DEBUG_BUILD

class tensor
{
	struct M
	{
#if TENSOR_USE_DATA_SPAN
		std::span<fp> data_span;
#endif

		tensor_dims dims;
		size_t capacity = 0;
		std::unique_ptr<fp[]> data = nullptr;

		void allocate(tensor_dims dims)
		{
			data = std::make_unique<fp[]>(dims.total());
			capacity = dims.total();
			this->dims = dims;
#if TENSOR_USE_DATA_SPAN
			data_span = { data.get(), dims.total() };
#endif
		}
		void resize(tensor_dims dims)
		{
			if (dims.total() > capacity)
				allocate(dims);
			else
				this->dims = dims;
		}
	} m;

	tensor(M&& m) : m(std::move(m)) {}

	template<class T>
	using nl = std::numeric_limits<T>;

	static constexpr idx_t _default_idx = nl<idx_t>::is_signed ? nl<idx_t>::min() + 1 : nl<idx_t>::max();
	static constexpr idx_t default_idx = DO_DEBUG_CHECKS ? _default_idx : 0;

	template<class R>
	static M init_from_range_with_dims(const R& r, tensor_dims dims)
	{
		xassert(dims.total() == std::size(r), "tensor::from_range: size mismatch");

		M m;
		m.allocate(dims);
		std::copy(std::begin(r), std::end(r), m.data.get());
		return m;
	}
	template<class R>
	static M init_from_range(const R& r)
	{
		return init_from_range_with_dims(r, std::size(r));
	}

	void internal_reset()
	{
		m = {};
	}


public:
	tensor() = default;
	tensor(const tensor& other)
		: m(init_from_range_with_dims(other, other.dims())) {}
	tensor(tensor&& o) noexcept
		: m(std::move(o.m))
	{
		o.internal_reset();
	}

	tensor& operator=(const tensor& other)
	{
		this->tensor::tensor(other);
		return *this;
	}
	tensor& operator=(tensor&& rhs) noexcept
	{
		this->m = std::move(rhs.m);
		rhs.internal_reset();
		return *this;
	}


	explicit tensor(size_t h, size_t w = 1, size_t d = 1)
		: tensor(tensor_dims{ h, w, d }) {}

	tensor(tensor_dims dims)
	{
		this->resize_storage(dims);
	}


	template<class R>
	static tensor from_range(const R& r)
	{
		return tensor{ init_from_range(r) };
	}
	static tensor from_range(std::initializer_list<fp> r)
	{
		return tensor{ init_from_range(r) };
	}


	template<class MapFn, class... Args>
	void map_from(const tensor& rhs, MapFn&& fn, Args&& ...args)
	{
		this->resize_storage(rhs.dims());
		for (size_t i = 0; i < rhs.size(); ++i)
			(*this)[i] = xinvoke(fn, args, rhs[i]);
	}

	void reshape(tensor_dims dims)
	{
		xassert(dims.total() == size(), "tensor::reshape: invalid new shape");
		m.dims = dims;
	}

	void resize_storage(tensor_dims dims)
	{
		m.resize(dims);
	}
	void resize_clear_storage(tensor_dims dims)
	{
		resize_storage(dims);
		zero_out();
	}
	void zero_out()
	{
		memset(data(), 0, size() * sizeof(fp));
	}
	void clear()
	{
		m.dims = {};
	}
	void deallocate()
	{
		internal_reset();
	}

	size_t size() const noexcept { return m.dims.total(); };

	auto dims() const noexcept { return m.dims; }

	auto data() noexcept { return m.data.get(); }
	auto data() const noexcept { return (const fp*)m.data.get(); }

	auto begin() { return data(); }
	auto begin() const { return data(); }
	auto end() { return begin() + size(); }
	auto end() const { return begin() + size(); }

	fp& operator()(idx_t y = default_idx, idx_t x = default_idx, idx_t z = default_idx)
	{
		return data()[to_linear_idx_with_checks(y, x, z)];
	}
	const fp& operator()(idx_t y = default_idx, idx_t x = default_idx, idx_t z = default_idx) const
	{
		return data()[to_linear_idx_with_checks(y, x, z)];
	}
	fp& operator[](idx_t idx)
	{
		check_no_overflow(idx, m.dims.total());
		return data()[idx];
	}
	const fp& operator[](idx_t idx) const
	{
		check_no_overflow(idx, m.dims.total());
		return data()[idx];
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
			xassert(idx >= 0 && (size_t)idx < dim, "tensor: invalid index {} for dim {}", idx, dim);
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

void multiply(tensor& left, fp scalar)
{
	for (auto& x : left)
		x *= scalar;
}
void multiply(const tensor& left, const tensor& right, tensor& out)
{
	xassert(left.dims().depth == 1 && right.dims().depth == 1, "3D multiplication not supported");
	xassert(left.dims().width == right.dims().height, "Tensors incompatible for multiplication");

	const size_t w = right.dims().width;
	const size_t h = left.dims().height;
	const size_t l = left.dims().width;
	out.resize_storage({ h, w, 1 });

	for (size_t y = 0; y < h; ++y)
		for (size_t k = 0; k < l; ++k)
			for (size_t x = 0; x < w; ++x)
				out(y, x) += left(y, k) * right(k, x);
}

void add(tensor& left, const tensor& right)
{
	xassert(left.dims() == right.dims(), "incompatible tensors");

	for (size_t i = 0; i < left.size(); ++i)
		left[i] += right[i];
}
void add(tensor& left, const tensor& right, fp scale, thread_pool& pool)
{
	xassert(left.dims() == right.dims(), "incompatible tensors");

	auto worker = [&, scale](size_t thread_id, size_t begin, size_t end) {
		for (size_t i = begin; i < end; ++i)
			left[i] += right[i] * scale;
	};
	pool.schedule_split_work(0, left.dims().total(), worker);
}

EXPORT_END
