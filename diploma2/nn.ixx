
module;

#include <variant>
#include <memory>
#include <filesystem>
#include <unordered_map>

#include <defs.hpp>

export module diploma.nn;
import diploma.lin_alg;
import diploma.utility;
import diploma.thread_pool;
import diploma.bmp;

import libksn.type_traits;





template<bool is_for_feed_back>
size_t calculate_convolution_output_dim(size_t in, size_t kernel)
{
	if constexpr (is_for_feed_back)
		return in + kernel - 1;
	else
		return in - kernel + 1;
}
template<bool is_for_feed_back>
tensor_dims calculate_convolution_output_dims(tensor_dims in, tensor_dims kernel)
{
	return {
		calculate_convolution_output_dim<is_for_feed_back>(in.width, kernel.width),
		calculate_convolution_output_dim<is_for_feed_back>(in.height, kernel.height),
		safe_div(kernel.depth, in.depth)
	};
}

export template<bool is_for_feed_back = false>
void perform_full_convolution(const tensor& input, const tensor& kernels, tensor& output)
{
	constexpr static bool is_for_feed_forward = !is_for_feed_back;

	const auto [kernel_height, kernel_width, kernel_count] = kernels.dims();
	const auto [input_height, input_width, input_images_count] = input.dims();
	const auto [out_width, out_height, out_images_count] =
		calculate_convolution_output_dims<is_for_feed_back>(input.dims(), kernels.dims());

	const size_t kernel_x_offset = is_for_feed_forward ? 0 : kernel_width - 1;
	const size_t kernel_y_offset = is_for_feed_forward ? 0 : kernel_height - 1;

	auto kernel_idx_projection = [](size_t x, size_t sz) {
		if constexpr (is_for_feed_forward)
			return x;
		else
			return sz - 1 - x;
	};
	auto kernel_at = [&](size_t y, size_t x, size_t kernel_id) {
		return kernels(
			kernel_idx_projection(y, kernel_height),
			kernel_idx_projection(x, kernel_width),
			kernel_id);
	};

	output.resize_clear_storage({ out_height, out_width, out_images_count });

	for (size_t out_image = 0; out_image < out_images_count; ++out_image)
		for (size_t in_image = 0; in_image < input_images_count; ++in_image)
		{
			const size_t kernel_id = out_image * input_images_count + in_image;

			auto do_output_element = [&](size_t y, size_t x) {
				const size_t n_begin = is_for_feed_forward ? 0 : std::max(kernel_x_offset, x) - x;
				const size_t m_begin = is_for_feed_forward ? 0 : std::max(kernel_y_offset, y) - y;
				const size_t n_end = is_for_feed_forward ? kernel_width : std::min(kernel_width, input_width + kernel_x_offset - x);
				const size_t m_end = is_for_feed_forward ? kernel_height : std::min(kernel_height, input_height + kernel_y_offset - y);

				fp val = 0;
				for (size_t m = m_begin; m < m_end; ++m)
					for (size_t n = n_begin; n < n_end; ++n)
						val += input(y + m - kernel_y_offset, x + n - kernel_x_offset, in_image) * kernel_at(m, n, kernel_id);

				output(y, x, out_image) += val;
			};

			for (size_t y = 0; y < out_height; ++y)
				for (size_t x = 0; x < out_width; ++x)
					do_output_element(y, x);
		}
}



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
		for (size_t i = 0; i < in.size(); ++i)
		{
			in[i] = exp(in[i]);
			normalizer += in[i];
		}
		for (auto& x : in)
			x /= normalizer;
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

		constexpr fp rng_range = (fp)2;

		data.resize_storage({ output_height, input_dims.height, 1 });
		randomize_range<fp>(data, -rng_range, rng_range);
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
		for (size_t i = 0; i < data.dims().height; ++i)
			for (size_t j = 0; j < data.dims().width; ++j)
				dLdparams(i, j) += dLdoutput[i] * input[j];
	}

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
		constexpr fp rng_range = (fp)2;

		data.resize_storage(input_dims);
		randomize_range(data, -rng_range, rng_range);

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

		constexpr fp rng_range = (fp)2;
		data.resize_storage({ kernel_height, kernel_width, input_dims.depth * out_images });
		randomize_range(data, -rng_range, rng_range);

		return { input_dims.height - kernel_height + 1, input_dims.width - kernel_width + 1, out_images };
	}

	void feed_forward(const tensor& in, tensor& out) const
	{
		perform_full_convolution(in, data, out);
	}
	void feed_back(const tensor&, const tensor&, const tensor& dLda, tensor& dLda_prev) const
	{
		perform_full_convolution<true>(dLda, data, dLda_prev);
	}
	void accumulate_gradient(const tensor& input, const tensor& dLdoutput, tensor& dLdparams) const
	{
		perform_full_convolution(input, dLdoutput, dLdparams);
	}

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
		constexpr fp rng_range = (fp)2;

		data.resize_storage(input_dims.depth);
		randomize_range(data, -rng_range, rng_range);

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
};

class pooling_layer
{
	friend class model;
	size_t pooling_width, pooling_height;

	tensor_dims init(tensor_dims input_dims) const
	{
		xassert(
			input_dims.width % pooling_width == 0 &&
			input_dims.height % pooling_height == 0,
			"Image size must be divisible by pooling size"
		);

		return { input_dims.height / pooling_height, input_dims.width / pooling_width, input_dims.depth };
	}

	void feed_forward(const tensor& in, tensor& out) const
	{
		//TODO: in-place?
		out.resize_storage({
			in.dims().height / pooling_height,
			in.dims().width / pooling_width,
			in.dims().depth }
			);

		for (size_t z = 0; z < out.dims().depth; ++z)
			for (size_t y = 0; y < out.dims().height; ++y)
				for (size_t x = 0; x < out.dims().width; ++x)
				{
					fp val = -std::numeric_limits<fp>::infinity();

					const size_t y_block = pooling_height * y;
					const size_t x_block = pooling_width * x;

					for (size_t n = 0; n < pooling_height; ++n)
						for (size_t m = 0; m < pooling_width; ++m)
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
					const fp& out_r = out(y, x, z);
					const fp& dLda_r = dLda(y, x, z);

					for (size_t n = 0; n < pooling_height; ++n)
						for (size_t m = 0; m < pooling_width; ++m)
						{
							const size_t y_block = pooling_height * y;
							const size_t x_block = pooling_width * x;
							const fp& in_r = in(y_block + n, x_block + m, z);
							fp& dLda_prev_r = dLda_prev(y_block + n, x_block + m, z);

							dLda_prev_r = (in_r == out_r) ? dLda_r : 0;
						}
				}
	}
	void accumulate_gradient(const tensor&, const tensor&, tensor&) const
	{
	}

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
};





class mse_loss_function
{
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
	static void f_update_helper(fp& accumulator, fp expected, fp observed)
	{
		using std::log;

		if (expected == 0)
			return;

		accumulator -= expected * log(observed);
	}
	static void df_update_helper(fp& accumulator, fp expected, fp observed)
	{
		using std::log;

		if (expected == 0)
			accumulator = 0;
		else
			accumulator = expected / observed;
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





class sgd_optimizer
{
	fp rate;
	size_t batch_size = SIZE_MAX;

public:
	sgd_optimizer(fp learning_rate, size_t max_batch_size = SIZE_MAX)
		: rate(learning_rate), batch_size(max_batch_size) {}
};






class model
{
	using layer_holder_t = std::variant<dense_layer, untied_bias_layer, convolution_layer, pooling_layer, tied_bias_layer, leaky_relu_layer, softmax_layer, flattening_layer>;
	using optimizer_holder_t = std::optional<std::variant<sgd_optimizer>>;
	using loss_function_holder_t = std::optional<std::variant<mse_loss_function, cross_entropy_loss_function>>;

#define variant_invoke(variant, method, ...) std::visit([&](auto&& stored_obj) { return stored_obj.method(__VA_ARGS__); }, variant)

	struct M
	{
		std::vector<layer_holder_t> layers;
		optimizer_holder_t optimizer;
		loss_function_holder_t loss_function;
		tensor_dims input_dims;
		tensor_dims output_dims;
	} m;

	friend class model_builder;

	bool has_optimizer() const
	{
		return m.optimizer.has_value();
	}
	bool is_finished() const
	{
		return this->has_optimizer();
	}
	void assert_is_finished(bool is = true) const
	{
		xassert(this->is_finished() == is, "model: action not applicable for {}finished models", this->is_finished() ? "" : "un");
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
	model(tensor_dims input_layer_dimensions)
		: m{ .input_dims = input_layer_dimensions, .output_dims = input_layer_dimensions } {}

	void add_layer(layer_holder_t layer)
	{
		assert_is_finished(false);
		m.output_dims = variant_invoke(layer, init, m.output_dims);
		m.layers.push_back(std::move(layer));
	}

	void finish(optimizer_holder_t optimizer, loss_function_holder_t loss_function)
	{
		assert_is_finished(false);
		xassert(m.layers.size() != 0, "model::finish: model must have at least one layer");

		m.optimizer = optimizer;
		m.loss_function = loss_function;
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
	size_t get_layer_parameter_count(size_t i) const noexcept
	{
		xassert(i < get_layer_count(), "Invalid layer index");
		return get_raw_layer_parameter_count(m.layers[i]).total();
	}
	size_t get_total_parameter_count() const noexcept
	{
		size_t result = 0;
		for (const auto& l : m.layers)
			result += get_raw_layer_parameter_count(l).total();
		return result;
	}

	template<class Dataset>
	void fit(const Dataset& dataset, thread_pool& pool)
	{
		assert_is_finished();

		struct thread_state { std::vector<tensor> dLdparams; };
		std::vector<thread_state> states(pool.size(), thread_state{ _create_layer_adjustment_vector() });

		//TODO: test new thread_pool API vs old one
		auto worker = [&](size_t thread_id, size_t begin, size_t end) {
			for (size_t i = begin; i < end; ++i)
			{
				const auto& [input, expected] = dataset[i];
				accumulate_gradient_single(input, expected, states[thread_id].dLdparams);
			}
		};
		pool.schedule_split_work(0, dataset.size(), worker);
		pool.barrier();

		//TODO: make use of stored optimizer object

		const fp scale = -0.05f / get_total_parameter_count() / dataset.size();

		for (size_t j = 0; j < states.size(); ++j)
		{
			for (size_t i = 0; i < m.layers.size(); ++i)
				adjust_layer_parameters(m.layers[i], states[j].dLdparams[i], scale, pool);
			pool.barrier();
		}

		//TODO: accumulate metrics along the way
	}

	template<class Dataset>
	fp evaluate(const Dataset& dataset, thread_pool& pool)
	{
		assert_is_finished();

		struct thread_state { fp loss{}; };
		std::vector<thread_state> states(pool.size());

		auto worker = [&](size_t thread_id, size_t begin, size_t end) {
			for (size_t i = begin; i < end; ++i)
			{
				const auto& [input, expected] = dataset[i];
				states[thread_id].loss += this->loss(this->predict(input), expected);
			}
		};
		pool.schedule_split_work(0, dataset.size(), worker);
		pool.barrier();

		fp sum = 0;
		for (auto state : states)
			sum += state.loss;
		return sum;
	}

	fp loss(const tensor& observed, const tensor& expected)
	{
		return variant_invoke(m.loss_function.value(), f, observed, expected);
	}
	fp loss_contribution(const tensor& input, const tensor& expected)
	{
		return loss(predict(input), expected);
	}

private:
	auto _create_layer_adjustment_vector()
	{
		std::vector<tensor> dLdparams(m.layers.size());
		for (size_t i = 0; i < m.layers.size(); ++i)
			dLdparams[i].resize_clear_storage(get_raw_layer_parameter_count(m.layers[i]));

		return dLdparams;
	}

	void accumulate_gradient_single(tensor in, const tensor& label, std::vector<tensor>& dLdparams)
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



class dataset_wrapper
{
	struct dataset_pair
	{
		tensor input;
		tensor expected;
	};

	using nstr = std::filesystem::path::string_type;

	std::vector<dataset_pair> dataset;
	std::vector<nstr> class_labels;

	void read_labels(cpath root)
	{
		for (const auto& entry : std::filesystem::directory_iterator(root))
		{
			xassert(entry.is_directory(), "dataset root must contain folders with labels as names");
			class_labels.push_back(entry.path().filename().native());
		}
	}
	void read_directory(cpath dir, const tensor& expected)
	{
		bmp_image image;
		for (const auto& entry : std::filesystem::directory_iterator(dir))
		{
			xassert(entry.is_regular_file(), "Extra object in dataset subdirectory {}", (char*)dir.generic_u8string().data());
			image.read(entry.path(), true);
			dataset.push_back({ std::move(image.data), expected});
		}
	}

public:
	dataset_wrapper(cpath root)
	{
		read_labels(root);

		tensor expected(class_labels.size());
		for (size_t idx = 0; idx < class_labels.size(); ++idx)
		{
			expected[idx] = 1;
			read_directory(root / class_labels[idx], expected);
			expected[idx] = 0;
		}
	}

	void shuffle() noexcept
	{
		static std::mt19937_64 rng(rng_seed);
		std::ranges::shuffle(dataset, rng);
	}

	const auto& operator[](idx_t n) const
	{
		xassert(n >= 0 && (size_t)n < dataset.size(), "dataset_wrapper::operator[]: invalid index {}", n);
		return dataset[n];
	}

	size_t size() const noexcept
	{
		return dataset.size();
	}

	tensor_dims input_size() const
	{
		xassert(!dataset.empty(), "Empty dataset");

		tensor_dims input_dims = dataset[0].input.dims();

		for (const auto& [input, _] : dataset)
			xassert(input.dims() == input_dims, "Inconsistent input size across dataset");

		return input_dims;
	}
};

EXPORT_END
