
module;
#include <immintrin.h>

export module diploma.nn;

import diploma.lin_alg;
import diploma.utility;
import diploma.thread_pool;
import diploma.image;
import diploma.dataset;
import diploma.serialization;
import diploma.conio;

import libksn.type_traits;


import <variant>;
import <memory>;
import <filesystem>;
import <unordered_map>;

import "defs.hpp";





template<bool outer_convolution>
size_t calculate_convolution_output_dim(size_t in, size_t kernel)
{
	if constexpr (outer_convolution)
		return in + kernel - 1;
	else
		return in - kernel + 1;
}
template<bool outer_convolution, bool as_product>
tensor_dims calculate_convolution_output_dims(tensor_dims in, tensor_dims kernel)
{
	return {
		calculate_convolution_output_dim<outer_convolution>(in.width, kernel.width),
		calculate_convolution_output_dim<outer_convolution>(in.height, kernel.height),
		as_product ? kernel.depth * in.depth : safe_div(kernel.depth, in.depth)
	};
}

template<size_t N>
using size_constant = std::integral_constant<size_t, N>;

#define repeat_call_arg2iota8(f, x) f(x, 1) f(x, 2) f(x, 3) f(x, 4) f(x, 5) f(x, 6) f(x, 7) f(x, 8)
#define repeat_call_arg2iota8_1(f, x) f(x, 1) f(x, 2) f(x, 3) f(x, 4) f(x, 5) f(x, 6) f(x, 7) f(x, 8)

template<size_t N>
constexpr __m256i get_load_mask()
{
	static_assert(N <= 8);
	static_assert(N > 0);
#define __f(i) ((i < N) ? -1 : 0)
	return _mm256_set_epi32(__f(7), __f(6), __f(5), __f(4), __f(3), __f(2), __f(1), __f(0));
}
template<size_t N>
consteval int get_mirror_permutation1()
{
	static_assert(N <= 4);
	static_assert(N > 0);

	int x = 0;
	for (int i = 0; i < (int)N; ++i)
	{
		x += (N - 1 - i) << (2 * i);
	}
	return x;
}
template<size_t N>
constexpr __m256i get_mirror_permutation2()
{
	static_assert(N <= 8);
	static_assert(N > 4);

#define __f(i) ((i < N) ? (N - 1 - i) : i)
	return _mm256_set_epi32(__f(7), __f(6), __f(5), __f(4), __f(3), __f(2), __f(1), __f(0));
}
export template<size_t N>
fp ymm_accumulate(__m256 ymm)
{
	//TODO: fix for N > 4
	static_assert(N >= 1 && N <= 8);
	if constexpr (N > 1) ymm = _mm256_hadd_ps(ymm, _mm256_setzero_ps());
	if constexpr (N > 2) ymm = _mm256_hadd_ps(ymm, _mm256_setzero_ps());
	if constexpr (N > 4) ymm = _mm256_add_ps(ymm, _mm256_castps128_ps256(_mm256_extractf32x4_ps(ymm, 1)));
	return _mm256_cvtss_f32(ymm);
}

export template<bool outer_convolution = false, bool as_product = false>
void perform_full_convolution(const tensor& input, const tensor& kernels, tensor& output)
{
	static_assert(!outer_convolution || !as_product);
	constexpr static bool inner_convolution = !outer_convolution;
	constexpr static bool mirror_kernel = outer_convolution;

	const auto [kernel_height, kernel_width, kernel_count] = kernels.dims();
	const auto [input_height, input_width, input_images_count] = input.dims();
	const auto [output_width, output_height, output_images_count] =
		calculate_convolution_output_dims<outer_convolution, as_product>(input.dims(), kernels.dims());

	const size_t kernel_x_offset = inner_convolution ? 0 : kernel_width - 1;
	const size_t kernel_y_offset = inner_convolution ? 0 : kernel_height - 1;

	auto kernel_idx_projection = [](size_t x, size_t sz) {
		if constexpr (mirror_kernel)
			return sz - 1 - x;
		else
			return x;
	};
	auto kernel_at = [&](size_t y, size_t x, size_t kernel_id) {
		return kernels(
			kernel_idx_projection(y, kernel_height),
			kernel_idx_projection(x, kernel_width),
			kernel_id
		);
	};

	output.resize_clear_storage({ output_height, output_width, output_images_count });

	size_t kernel_id = 0, input_image_id = 0, output_image_id = 0;
	
	while (true)
	{
		//convolution strategy:

		auto do_output_element = [&]<bool fully_in_bounds>(std::bool_constant<fully_in_bounds>, size_t output_y, size_t output_x)
		{
			auto get_coord_bounds = [](size_t input_size, size_t output_coord, size_t kernel_offset, size_t kernel_size) -> std::pair<size_t, size_t> {
				if constexpr (fully_in_bounds)
				{
					return { 0, kernel_size };
				}
				else
				{
					const size_t min = std::max<size_t>(kernel_offset, output_coord) - output_coord;
					const size_t max = std::min<size_t>(input_size + kernel_offset - output_coord, kernel_size);
					return { min, max };
				}
			};

			auto& out = output(output_y, output_x, output_image_id);

			auto apply_kernel_default = [&] {
				const auto [n_begin, n_end] = get_coord_bounds(input_height, output_y, kernel_y_offset, kernel_height);
				const auto [m_begin, m_end] = get_coord_bounds(input_width, output_x, kernel_x_offset, kernel_width);

				fp val = 0;
				for (size_t n = n_begin; n < n_end; ++n)
				{
					for (size_t m = m_begin; m < m_end; ++m)
					{
						const size_t in_y = output_y + n - kernel_y_offset;
						const size_t in_x = output_x + m - kernel_x_offset;
						val += input(in_y, in_x, input_image_id) * kernel_at(n, m, kernel_id);
					}
				}
				out += val;
			};

			if constexpr (!fully_in_bounds)
				apply_kernel_default();
			else
			{
				auto apply_kernel_simd = [&]<size_t kh, size_t kw>(size_constant<kh>, size_constant<kw>)
				{
					const float* pk = &kernels(mirror_kernel ? kernel_height - 1 : 0, 0, kernel_id);
					const float* pi = &input(output_y - kernel_y_offset, output_x - kernel_x_offset, input_image_id);

					const auto load_mask = get_load_mask<kw>();
					const auto mirror_permutation = [] { if constexpr (!mirror_kernel) return 0; else if constexpr (kw <= 4) return get_mirror_permutation1<kw>(); else return get_mirror_permutation2<kw>(); }();

					__m256 accumulator = _mm256_setzero_ps();
					__m256 kernel_data[kh];
#define advance_kernel() if constexpr(mirror_kernel) pk -= kernel_width; else pk += kernel_width;
#define maybe_load_kernel_row(_, i) if constexpr(i - 1 < kh) { kernel_data[i - 1] = _mm256_maskload_ps(pk, load_mask); advance_kernel(); }
#define mirror_row4(i) { kernel_data[i - 1] = _mm256_permute_ps(kernel_data[i - 1], mirror_permutation); }
#define mirror_row8(i) { kernel_data[i - 1] = _mm256_permutevar8x32_ps(kernel_data[i - 1], mirror_permutation); }
#define maybe_mirror_row(_, i) if constexpr(i - 1 < kh && mirror_kernel) { if constexpr (kw <= 4) mirror_row4(i) else mirror_row8(i); }
#define maybe_accumulate(_, i) if constexpr(i - 1 < kh) accumulator = _mm256_fmadd_ps(kernel_data[i - 1], _mm256_loadu_ps(pi + (i - 1) * input_width), accumulator);
					repeat_call_arg2iota8(maybe_load_kernel_row, _);
					repeat_call_arg2iota8(maybe_mirror_row, _);
					repeat_call_arg2iota8(maybe_accumulate, _);

					out += ymm_accumulate<kw>(accumulator);
				};

#define cover_kernel_size(w, h) else if (kernel_width == (w) && kernel_height == (h)) { apply_kernel_simd(size_constant<(h)>{}, size_constant<(w)>{}); }

				if (false) {}
				repeat_call_arg2iota8(repeat_call_arg2iota8_1, cover_kernel_size)
				else apply_kernel_default();
			}
		};

		auto kernel_fully_in_bounds_1d = [](size_t input_size, size_t output_coord, size_t kernel_offset, size_t kernel_size) {
			if constexpr (inner_convolution)
				return true;
			else
				return output_coord >= kernel_offset && output_coord + kernel_size < input_size + kernel_offset + 1;
		};

		for (size_t output_y = 0; output_y < output_height; ++output_y)
		{
			const bool in_bounds_y = kernel_fully_in_bounds_1d(input_height, output_y, kernel_y_offset, kernel_height);

			for (size_t output_x = 0; output_x < output_width; ++output_x)
			{
				const bool fully_in_bounds = in_bounds_y && kernel_fully_in_bounds_1d(input_width, output_x, kernel_x_offset, kernel_width);
				
				if (fully_in_bounds)
					do_output_element(std::true_type{}, output_y, output_x);
				else
					do_output_element(std::false_type{}, output_y, output_x);
			}
		}
		

		//index progression strategy:

		if constexpr (as_product)
		{
			if (++output_image_id == output_images_count)
				break;
			if (++input_image_id == input_images_count)
			{
				input_image_id = 0;
				++kernel_id;
			}
		}
		else
		{
			if (++input_image_id == input_images_count)
				{ input_image_id = 0; output_image_id++; }
			if (output_image_id == output_images_count)
				break;
			kernel_id = output_image_id * input_images_count + input_image_id;
		}
	}
}



constexpr fp rng_range = (fp)0.2;

enum serialized_obj_type
{
	none = 0,
	layer,
	loss_function,
};

enum class layer_type : uint32_t
{
	none = 0,
	leaky_relu,
	softmax,
	dense,
	untied_bias,
	convolution,
	tied_bias,
	pooling,
	flattening,
};

enum class loss_function_type
{
	none = 0,
	mse,
	cross_entropy,
};

EXPORT_BEGIN

class leaky_relu_layer
{
	friend class model;

	fp slope;

	fp activate(fp x) const
	{
		if (x < 0)
			return slope * x;
		else
			return x;
	}

	tensor_dims init(tensor_dims input_dims) const
	{
		return input_dims;
	}
	void feed_forward(tensor& in, tensor& out) const
	{
		for (auto& x : in)
			x = activate(x);
		std::swap(in, out);
	}
	void feed_back(const tensor& in, const tensor&, tensor& dLda, tensor& dLda_prev) const
	{
		for (size_t i = 0; i < in.size(); ++i)
			if (in[i] < 0)
				dLda[i] *= slope;

		std::swap(dLda, dLda_prev);
	}
	void accumulate_gradient(const tensor&, const tensor&, tensor&) const
	{
	}

	void serialize(serializer_t& serializer) const
	{
		serializer(serialized_obj_type::layer);
		serializer(layer_type::leaky_relu);
		serializer(slope);
	}
	void deserialize(deserializer_t& deserializer)
	{
		deserializer(slope);
	}

public:
	leaky_relu_layer(fp negative_side_slope = 0.1)
		: slope(negative_side_slope) {}
};

class softmax_layer
{
	friend class model;

	tensor_dims init(tensor_dims input_dims) const
	{
		return input_dims;
	}

	void feed_forward(tensor& in, tensor& out) const
	{
		fp normalizer = 0;
		fp max_v = -INFINITY;
		size_t max_idx = 0;
		for (size_t i = 0; i < in.size(); ++i)
		{
			if (in[i] > max_v)
			{
				max_v = in[i];
				max_idx = i;
			}
			in[i] = exp(in[i]);
			normalizer += in[i];
		}
		if (normalizer != INFINITY)
		{
			for (auto& x : in)
				x /= normalizer;
		}
		else
		{
			for (auto& x : in) x = 0;
			in[max_idx] = 1;
		}
		std::swap(in, out);
	}
	void feed_back(const tensor&, const tensor& out, tensor& dLda, tensor& dLda_prev) const
	{
		fp helper_sum = 0;
		for (size_t i = 0; i < out.size(); ++i)
			helper_sum += dLda[i] * out[i];

		for (size_t i = 0; i < dLda.size(); ++i)
			dLda[i] = out[i] * (dLda[i] - helper_sum);

		std::swap(dLda, dLda_prev);
	}
	void accumulate_gradient(const tensor&, const tensor&, tensor&) const
	{

	}

	void serialize(serializer_t& serializer) const
	{
		serializer(serialized_obj_type::layer);
		serializer(layer_type::softmax);
	}
	void deserialize(deserializer_t& deserializer)
	{
	}
};



class dense_layer
{
	friend class model;

	size_t output_height;
	tensor data;

	tensor_dims init(tensor_dims input_dims)
	{
		xassert(output_height != 0, "Dense layer output size must not be 0");
		xassert(input_dims.width == 1 && input_dims.depth == 1, "Dense layer takes Nx1x1 tensor as input");

		data.resize_storage({ output_height, input_dims.height, 1 });
		randomize_range(data, -rng_range, rng_range);
		return { output_height, 1, 1 };
	}

	void feed_forward(const tensor& in, tensor& out) const
	{
		multiply(data, in, out);
	}
	void feed_back(const tensor&, const tensor&, const tensor& dLda, tensor& dLda_prev) const
	{
		dLda_prev.resize_clear_storage(data.dims().width);
		for (size_t j = 0; j < dLda.size(); ++j)
			for (size_t k = 0; k < dLda_prev.size(); ++k)
				dLda_prev[k] += dLda[j] * data(j, k);
	}
	void accumulate_gradient(const tensor& input, const tensor& dLdoutput, tensor& dLdparams) const
	{
		dLdparams.resize_storage(data.dims());
		for (size_t i = 0; i < data.dims().height; ++i)
			for (size_t j = 0; j < data.dims().width; ++j)
				dLdparams(i, j) += dLdoutput[i] * input[j];
	}

	void serialize(serializer_t& serializer) const
	{
		serializer(serialized_obj_type::layer);
		serializer(layer_type::dense);
		serializer(output_height);
		serializer(data);
	}
	void deserialize(deserializer_t& deserializer)
	{
		deserializer(output_height);
		deserializer(data);
	}
	dense_layer() = default;

public:
	dense_layer(size_t output_size)
		: output_height(output_size) {}
};

class untied_bias_layer
{
	friend class model;

	tensor data;

	tensor_dims init(tensor_dims input_dims)
	{
		data.resize_storage(input_dims);

		return input_dims;
	}

	void feed_forward(tensor& in, tensor& out) const
	{
		for (size_t i = 0; i < in.size(); ++i)
			in[i] += data[i];
		std::swap(in, out);
	}
	void feed_back(const tensor&, const tensor&, tensor& dLda, tensor& dLda_prev) const
	{
		std::swap(dLda, dLda_prev);
	}
	void accumulate_gradient(const tensor&, const tensor& dLdoutput, tensor& dLdparams) const
	{
		dLdparams = dLdoutput;
	}

	void serialize(serializer_t& serializer) const
	{
		serializer(serialized_obj_type::layer);
		serializer(layer_type::untied_bias);
		serializer(data);
	}
	void deserialize(deserializer_t& deserializer)
	{
		deserializer(data);
	}
};



class convolution_layer
{
	friend class model;

	tensor data;

	size_t kernel_height, kernel_width;
	size_t out_images;

	tensor_dims init(tensor_dims input_dims)
	{
		xassert(kernel_width && kernel_height, "Convolution kernels must be non-zero in size");
		xassert(input_dims.width >= kernel_width && input_dims.height >= kernel_height, "Convolution kernels must not exceed the input in dimensions");

		data.resize_storage({ kernel_height, kernel_width, input_dims.depth * out_images });
		randomize_range(data, -rng_range, rng_range);

		return { input_dims.height - kernel_height + 1, input_dims.width - kernel_width + 1, out_images };
	}

	void feed_forward(const tensor& in, tensor& out) const
	{
		perform_full_convolution<false, false>(in, data, out);
	}
	void feed_back(const tensor&, const tensor&, const tensor& dLda, tensor& dLda_prev) const
	{
		perform_full_convolution<true, false>(dLda, data, dLda_prev);
	}
	void accumulate_gradient(const tensor& input, const tensor& dLdoutput, tensor& dLdparams) const
	{
		perform_full_convolution<false, true>(input, dLdoutput, dLdparams);
	}

	void serialize(serializer_t& serializer) const
	{
		serializer(serialized_obj_type::layer);
		serializer(layer_type::convolution);
		serializer(kernel_height);
		serializer(kernel_width);
		serializer(out_images);
		serializer(data);
	}
	void deserialize(deserializer_t& deserializer)
	{
		deserializer(kernel_height);
		deserializer(kernel_width);
		deserializer(out_images);
		deserializer(data);
	}

	convolution_layer() = default;

public:
	convolution_layer(size_t height, size_t width, size_t output_images)
		: kernel_height(height), kernel_width(width), out_images(output_images) {}
};

class tied_bias_layer
{
	friend class model;

	tensor data;

	tensor_dims init(tensor_dims input_dims)
	{
		data.resize_storage(input_dims.depth);

		return input_dims;
	}

	void feed_forward(tensor& in, tensor& out) const
	{
		for (size_t i = 0; i < in.size(); ++i)
			in[i] += data[i / in.dims().image_size()];
		std::swap(in, out);
	}
	void feed_back(const tensor&, const tensor&, tensor& dLda, tensor& dLda_prev) const
	{
		std::swap(dLda, dLda_prev);
	}
	void accumulate_gradient(const tensor&, const tensor& dLdoutput, tensor& dLdparams) const
	{
		dLdparams.resize_storage(data.dims());
		for (size_t k = 0; k < data.size(); ++k)
		{
			fp val = 0;
			for (size_t j = 0; j < dLdoutput.dims().height; ++j)
				for (size_t i = 0; i < dLdoutput.dims().width; ++i)
					val += dLdoutput(j, i, k);
			dLdparams[k] = val;
		}
	}

	void serialize(serializer_t& serializer) const
	{
		serializer(serialized_obj_type::layer);
		serializer(layer_type::tied_bias);
		serializer(data);
	}
	void deserialize(deserializer_t& deserializer)
	{
		deserializer(data);
	}
};

class pooling_layer
{
	friend class model;

	size_t pooling_width, pooling_height;
	tensor_dims output_dims;

	static size_t align_div(size_t a, size_t b)
	{
		return a / b + bool(a % b);
	}

	tensor_dims init(tensor_dims input_dims)
	{
		return output_dims = { align_div(input_dims.height, pooling_height), align_div(input_dims.width, pooling_width), input_dims.depth };
	}

	void feed_forward(const tensor& in, tensor& out) const
	{
		//TODO: in-place?
		out.resize_storage(output_dims);

		for (size_t z = 0; z < out.dims().depth; ++z)
			for (size_t y = 0; y < out.dims().height; ++y)
				for (size_t x = 0; x < out.dims().width; ++x)
				{
					fp val = -std::numeric_limits<fp>::infinity();

					const size_t y_block = pooling_height * y;
					const size_t x_block = pooling_width * x;

					const size_t n_max = std::min(pooling_height, in.dims().height - y_block);
					const size_t m_max = std::min(pooling_width, in.dims().width - x_block);

					for (size_t n = 0; n < n_max; ++n)
						for (size_t m = 0; m < m_max; ++m)
							val = std::max(val, in(y_block + n, x_block + m, z));

					out(y, x, z) = val;
				}
	}

	void feed_back(const tensor& in, const tensor& out, tensor& dLda, tensor& dLda_prev) const
	{
		dLda_prev.resize_storage(in.dims());

		for (size_t z = 0; z < out.dims().depth; ++z)
			for (size_t y = 0; y < out.dims().height; ++y)
				for (size_t x = 0; x < out.dims().width; ++x)
				{
					const size_t y_block = pooling_height * y;
					const size_t x_block = pooling_width * x;

					const size_t n_max = std::min(pooling_height, in.dims().height - y_block);
					const size_t m_max = std::min(pooling_width, in.dims().width - x_block);

					const fp& out_r = out(y, x, z);
					const fp& dLda_r = dLda(y, x, z);

					for (size_t n = 0; n < n_max; ++n)
						for (size_t m = 0; m < m_max; ++m)
						{
							const fp& in_r = in(y_block + n, x_block + m, z);
							fp& dLda_prev_r = dLda_prev(y_block + n, x_block + m, z);

							dLda_prev_r = (in_r == out_r) ? dLda_r : 0;
						}
				}
	}
	void accumulate_gradient(const tensor&, const tensor&, tensor&) const
	{
	}

	void serialize(serializer_t& serializer) const
	{
		serializer(serialized_obj_type::layer);
		serializer(layer_type::pooling);
		serializer(pooling_height);
		serializer(pooling_width);
		serializer(output_dims);
	}
	void deserialize(deserializer_t& deserializer)
	{
		deserializer(pooling_height);
		deserializer(pooling_width);
		deserializer(output_dims);
	}

	pooling_layer() = default;

public:
	pooling_layer(size_t pooling_height, size_t pooling_width)
		: pooling_height(pooling_height), pooling_width(pooling_width) {}
};

class flattening_layer
{
	friend class model;
	tensor_dims in_dims;

	tensor_dims init(tensor_dims input_dims)
	{
		in_dims = input_dims;
		return { input_dims.total() };
	}

	void feed_forward(tensor& in, tensor& out) const
	{
		in.reshape(in.dims().total());
		std::swap(in, out);
	}
	void feed_back(const tensor&, const tensor&, tensor& dLda, tensor& dLda_prev) const
	{
		dLda.reshape(in_dims);
		std::swap(dLda, dLda_prev);
	}
	void accumulate_gradient(const tensor&, const tensor&, tensor&) const
	{
	}

	void serialize(serializer_t& serializer) const
	{
		serializer(serialized_obj_type::layer);
		serializer(layer_type::flattening);
		serializer(in_dims);
	}
	void deserialize(deserializer_t& deserializer)
	{
		deserializer(in_dims);
	}
};





class mse_loss_function
{
	friend class model;

	void serialize(serializer_t& serializer) const
	{
		serializer(serialized_obj_type::loss_function);
		serializer(loss_function_type::mse);
	}

public:
	fp f(const tensor& observed, const tensor& expected)
	{
		xassert(observed.dims() == expected.dims(), "MSE::f: incompatible-shaped tensor inputs");

		fp result = 0;
		for (size_t i = 0; i < observed.size(); ++i)
		{
			fp diff = observed[i] - expected[i];
			result += diff * diff;
		}
		return result;
	}
	void df(const tensor& observed, const tensor& expected, tensor& result)
	{
		xassert(observed.dims() == expected.dims(), "MSE::df: incompatible-shaped tensor inputs");
		const size_t n = observed.size();

		result.resize_storage(n);
		for (size_t i = 0; i < n; ++i)
			result[i] = 2 * (observed[i] - expected[i]);
	}
};

class cross_entropy_loss_function
{
	friend class model;

	void serialize(serializer_t& serializer) const
	{
		serializer(serialized_obj_type::loss_function);
		serializer(loss_function_type::cross_entropy);
	}

	static constexpr fp epsilon = 1e-5f;

	static void f_update_helper(fp& accumulator, fp expected, fp observed)
	{
		using std::log;

		if (expected == 0)
			return;
		if (observed == 0)
			observed += epsilon;

		accumulator -= expected * log(observed);
	}
	static void df_update_helper(fp& accumulator, fp expected, fp observed)
	{
		using std::log;

		if (expected == 0)
			accumulator = 0;
		else
		{
			if (observed == 0)
				observed += epsilon;
			accumulator = expected / observed;
		}
	}
public:
	fp f(const tensor& observed, const tensor& expected)
	{
		xassert(observed.dims() == expected.dims(), "CE::f: incompatible-shaped tensor inputs");

		fp result = 0;
		for (size_t i = 0; i < observed.size(); ++i)
		{
			f_update_helper(result, expected[i], observed[i]);
			f_update_helper(result, 1 - expected[i], 1 - observed[i]);
		}
		return result / observed.size();
	}
	void df(const tensor& observed, const tensor& expected, tensor& result)
	{
		xassert(observed.dims() == expected.dims(), "CE::f: incompatible-shaped tensor inputs");

		result.resize_storage(observed.dims());
		for (size_t i = 0; i < result.size(); ++i)
		{
			fp a, b;
			df_update_helper(a, 1 - expected[i], 1 - observed[i]);
			df_update_helper(b, expected[i], observed[i]);
			result[i] = (a - b) / result.size();
		}
	}
};





class model
{
	using layer_holder_t = std::variant<dense_layer, untied_bias_layer, convolution_layer, pooling_layer, tied_bias_layer, leaky_relu_layer, softmax_layer, flattening_layer>;
	using loss_function_holder_t = std::optional<std::variant<mse_loss_function, cross_entropy_loss_function>>;

#define variant_invoke(variant, method, ...) std::visit([&](auto&& stored_obj) { return stored_obj.method(__VA_ARGS__); }, variant)

	struct M
	{
		std::vector<layer_holder_t> layers;
		loss_function_holder_t loss_function;
		tensor_dims input_dims;
		tensor_dims output_dims;
	} m;

	bool has_loss_function() const
	{
		return m.loss_function.has_value();
	}
	bool is_finished() const
	{
		return this->has_loss_function();
	}
	void assert_is_finished(bool is = true) const
	{
		xassert(
			this->is_finished() == is,
			"model: action not applicable for {}finished models",
			this->is_finished() ? "" : "un"
		);
	}

	bool has_input_dims() const
	{
		return m.input_dims != tensor_dims{};
	}
	void assert_has_input_dims(bool is = true) const
	{
		xassert(
			this->has_input_dims() == is,
			"model: action not applicable for model with{} input size specified",
			this->has_input_dims() ? "" : "out"
		);
	}

	//one must imagine the fun of using "#define concept static constexpr bool" to allow defining concepts inside classes
	template<class LayerType>
	static constexpr bool has_tensor_data_member =
		requires(LayerType layer) { { layer.data } -> ksn::same_to_cvref<tensor>; };

	static tensor_dims get_raw_layer_parameter_count(const layer_holder_t& layer)
	{
		auto visiter = [](const auto& obj) -> tensor_dims
		{
			if constexpr (has_tensor_data_member<decltype(obj)>)
				return obj.data.dims();
			else
				return {};
		};
		return std::visit(visiter, layer);
	}

	static void adjust_layer_parameters(layer_holder_t& layer, const tensor& adjustment, fp scale, thread_pool& pool)
	{
		dassert(get_raw_layer_parameter_count(layer) == adjustment.dims());

		auto visiter = [&](auto& obj)
		{
			if constexpr (has_tensor_data_member<decltype(obj)>)
				add(obj.data, adjustment, scale, pool);
		};
		std::visit(visiter, layer);
	}

public:
	model() {}

	model(tensor_dims input_layer_dimensions)
		: m{ .input_dims = input_layer_dimensions, .output_dims = input_layer_dimensions } {}

	void set_input_size(tensor_dims dims)
	{
		assert_has_input_dims(false);
		m.input_dims = m.output_dims = dims;
		assert_has_input_dims(true);
	}

	void add_layer(layer_holder_t layer)
	{
		assert_has_input_dims();
		assert_is_finished(false);
		m.output_dims = variant_invoke(layer, init, m.output_dims);
		m.layers.push_back(std::move(layer));
	}

	void finish(loss_function_holder_t loss_function)
	{
		assert_is_finished(false);
		xassert(m.layers.size() != 0, "model::finish: model must have at least one layer");

		m.loss_function = loss_function;
	}

	void reset()
	{
		m = {};
	}

	tensor predict(tensor in) const
	{
		assert_is_finished();
		xassert(in.dims() == m.input_dims, "Invalid model input size");

		tensor tmp;
		for (const auto& layer : m.layers)
		{
			variant_invoke(layer, feed_forward, in, tmp);
			std::swap(in, tmp);
		}
		return in;
	}

	size_t get_layer_count() const noexcept
	{
		return m.layers.size();
	}
	size_t get_total_parameter_count() const noexcept
	{
		size_t result = 0;
		for (const auto& l : m.layers)
			result += get_raw_layer_parameter_count(l).total();
		return result;
	}

	struct model_statistics
	{
		fp loss;
		fp accuracy;

	private:
		friend class model;

		void merge(const model_statistics& other)
		{
			loss += other.loss;
			accuracy += other.accuracy;
		}
		void normalize(size_t observations)
		{
			loss /= observations;
			accuracy /= observations;
		}
	};

	template<dataset_wrapper Dataset>
	model_statistics fit(Dataset& dataset, thread_pool& pool, fp learning_rate, size_t batch_size = SIZE_MAX, bool should_report_progress = false)
	{
		assert_is_finished();

		batch_size = std::min(batch_size, size(dataset));
		shuffle(dataset, batch_size);

		struct thread_state
		{
			std::vector<tensor> dLdparams;
			model_statistics stats{};
			bool used = false;
		};
		std::vector<thread_state> states(pool.size(), { std::vector<tensor>{ m.layers.size() }});

		cursor_pos_holder cursor;
		//TODO: test new thread_pool API vs old one
		auto worker = [&](size_t thread_id, size_t begin, size_t end) {
			auto report = [&](size_t n) { if (should_report_progress && thread_id == 0) { cursor.restore(); std::print("fit: {}/{}", n, end - begin); }};

			states[thread_id].used = true;

			report(0);
			for (size_t i = begin; i < end; ++i)
			{
				const auto& [input, expected] = at(dataset, i);
				accumulate_gradient_single(input, expected, states[thread_id].dLdparams, states[thread_id].stats);
				report(i + 1);
			}
		};
		pool.schedule_split_work(0, batch_size, worker);
		pool.barrier();

		const fp scale = -learning_rate / batch_size;

		for (size_t j = 0; j < states.size(); ++j)
		{
			if (!states[j].used)
				continue;
			for (size_t i = 0; i < m.layers.size(); ++i)
				adjust_layer_parameters(m.layers[i], states[j].dLdparams[i], scale, pool);
			pool.barrier();
		}

		return merge_stats(states, batch_size);
	}

	template<dataset_wrapper Dataset>
	model_statistics evaluate(const Dataset& dataset, thread_pool& pool)
	{
		assert_is_finished();

		struct thread_state { model_statistics stats{}; };
		std::vector<thread_state> states(pool.size());

		auto worker = [&](size_t thread_id, size_t begin, size_t end) {
			for (size_t i = begin; i < end; ++i)
			{
				auto& state = states[thread_id];
				const auto& [input, expected] = at(dataset, i);
				const auto observed = this->predict(input);
				update_statistics(state.stats, observed, expected);
			}
		};
		pool.schedule_split_work(0, size(dataset), worker);
		pool.barrier();

		return merge_stats(states, size(dataset));
	}

	void serialize(serializer_t& serializer) const
	{
		assert_is_finished();

		serializer(m.input_dims);
		serializer(m.output_dims);
		for (const auto& layer : m.layers)
			variant_invoke(layer, serialize, serializer);
		variant_invoke(m.loss_function.value(), serialize, serializer);

		serializer.write_crc();
	}

	bool deserialize(deserializer_t& deserializer)
	{
		assert_has_input_dims(false);

		bool ok = true;
		bool stop = false;

		ok &= deserializer(m.input_dims);
		ok &= deserializer(m.output_dims);

		serialized_obj_type next_object_type;
		while (ok && !stop)
		{
			ok &= deserializer(next_object_type);

			if (!ok) 	break;

			switch (next_object_type)
			{
			case serialized_obj_type::layer:
				ok = try_deserialize_layer(deserializer);
				break;

			case serialized_obj_type::loss_function:
				ok = try_deserialize_loss_function(deserializer);
				stop = true;
				break;

			default:
				ok = false;
				break;
			}
		}

		if (ok) ok = deserializer.test_crc();
		if (!ok) reset();

		return ok;
	}

private:
	bool try_deserialize_layer(deserializer_t& deserializer)
	{
		std::optional<layer_holder_t> layer;

		layer_type type{};
		deserializer(type);
		switch (type)
		{
		case layer_type::leaky_relu: layer = leaky_relu_layer{}; break;
		case layer_type::softmax: layer = softmax_layer{}; break;
		case layer_type::dense: layer = dense_layer{}; break;
		case layer_type::untied_bias: layer = untied_bias_layer{}; break;
		case layer_type::convolution: layer = convolution_layer{}; break;
		case layer_type::tied_bias: layer = tied_bias_layer{}; break;
		case layer_type::pooling: layer = pooling_layer{}; break;
		case layer_type::flattening: layer = flattening_layer{}; break;
		default: break;
		}
		if (!layer.has_value())
			return false;

		variant_invoke(layer.value(), deserialize, deserializer);
		if (!deserializer)
			return false;

		m.layers.push_back(std::move(layer.value()));
		return true;
	}
	bool try_deserialize_loss_function(deserializer_t& deserializer)
	{
		loss_function_holder_t func;

		loss_function_type type{};
		deserializer(type);

		switch (type)
		{
		case loss_function_type::mse: func = mse_loss_function{}; break;
		case loss_function_type::cross_entropy: func = cross_entropy_loss_function{}; break;
		default: break;
		}

		if (!func.has_value())
			return false;

		m.loss_function = std::move(func);
		return true;
	}

	void update_statistics(model_statistics& stats, const tensor& observed, const tensor& expected)
	{
		stats.loss += this->loss(observed, expected);
		stats.accuracy += get_max_idx(observed) == get_max_idx(expected);
	}

	size_t get_layer_parameter_count(size_t i) const noexcept
	{
		xassert(i < get_layer_count(), "Invalid layer index");
		return get_raw_layer_parameter_count(m.layers[i]).total();
	}

	fp loss(const tensor& observed, const tensor& expected)
	{
		return variant_invoke(m.loss_function.value(), f, observed, expected);
	}
	fp loss_contribution(const tensor& input, const tensor& expected)
	{
		return loss(predict(input), expected);
	}

	template<class R>
	model_statistics merge_stats(const R& states, size_t normalizer)
	{
		model_statistics result{};
		for (auto state : states)
			result.merge(state.stats);
		result.normalize(normalizer);
		return result;
	}

	void accumulate_gradient_single(tensor in, const tensor& label, std::vector<tensor>& dLdparams, model_statistics& stats)
	{
		std::vector<tensor> activations;
		tensor tmp;
		for (size_t i = 0; i < m.layers.size(); ++i)
		{
			activations.push_back(in);
			variant_invoke(m.layers[i], feed_forward, in, tmp);
			std::swap(in, tmp);
		}
		activations.push_back(std::move(in));

		update_statistics(stats, activations.back(), label);

		tensor dLdactivations, dLdactivations_prev;
		variant_invoke(m.loss_function.value(), df, activations.back(), label, dLdactivations);

		for (size_t i = m.layers.size() - 1; true; --i)
		{
			variant_invoke(m.layers[i], accumulate_gradient, activations[i], dLdactivations, dLdparams[i]);
			if (i == 0) break;
			variant_invoke(m.layers[i], feed_back, activations[i], activations[i + 1], dLdactivations, dLdactivations_prev);
			std::swap(dLdactivations, dLdactivations_prev);
		}
	}
};

EXPORT_END
