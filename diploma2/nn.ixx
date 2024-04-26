
module;

#include <variant>
#include <memory>

#include <defs.hpp>

export module diploma.nn;
import diploma.lin_alg;
import diploma.utility;
import diploma.thread_pool;





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

	void feed_forward(tensor& in, tensor& out) const
	{
		for (auto& x : in)
			x = activate(x);
		std::swap(in, out);
	}

	tensor_dims init(tensor_dims input_dims)
	{
		return input_dims;
	}

public:
	leaky_relu_layer(fp negative_side_slope)
		: slope(negative_side_slope) {}
};

class softmax_layer
{
	friend class model;

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

	tensor_dims init(tensor_dims input_dims)
	{
		return input_dims;
	}
};



class dense_layer
{
	friend class model;

	size_t output_height;
	tensor data;

	void feed_forward(const tensor& in, tensor& out) const
	{
		multiply(data, in, out);
	}

	tensor_dims init(tensor_dims input_dims)
	{
		xassert(output_height != 0, "Dense layer output size must not be 0");
		xassert(input_dims.width == 1 && input_dims.depth == 1, "Dense layer takes Nx1x1 tensor as input");

		constexpr fp rng_range = (fp)0.1;

		data.resize_storage({ output_height, input_dims.height, 1 });
		randomize_range<fp>(data, -rng_range, rng_range);
		return { output_height, 1, 1 };
	}

public:
	dense_layer(size_t output_size)
		: output_height(output_size) {}
};

class untied_bias_layer
{
	friend class model;

	tensor data;

	void feed_forward(tensor& in, tensor& out) const
	{
		for (size_t i = 0; i < in.size(); ++i)
			in[i] += data[i];
		std::swap(in, out);
	}

	tensor_dims init(tensor_dims input_dims)
	{
		constexpr fp rng_range = (fp)0.1;

		data.resize_storage(input_dims);
		randomize_range(data, -rng_range, rng_range);

		return input_dims;
	}
};

void perform_full_convolution(const tensor& in, const tensor& kernels, tensor& out)
{
	const size_t in_images = in.dims().depth;
	const size_t out_images = kernels.dims().depth / in_images;

	const size_t out_width = in.dims().width - kernels.dims().width + 1;
	const size_t out_height = in.dims().height - kernels.dims().height + 1;

	const size_t kernel_width = kernels.dims().width;
	const size_t kernel_height = kernels.dims().height;

	//out.resize_storage({ out_height, out_width, out_images });
	out.resize_clear_storage({ out_height, out_width, out_images });

	for (size_t out_image = 0; out_image < out_images; ++out_image)
		for (size_t in_image = 0; in_image < in_images; ++in_image)
		{
			const size_t kernel_id = out_image * in_images + in_image;

			for (size_t y = 0; y < out_height; ++y)
				for (size_t x = 0; x < out_width; ++x)
				{
					fp val = 0;

					for (size_t n = 0; n < kernel_height; ++n)
						for (size_t m = 0; m < kernel_width; ++m)
							val += in(y + n, x + m, in_image) * kernels(n, m, kernel_id);

					out(y, x, out_image) += val;
				}
		}
}

class convolution_layer
{
	friend class model;

	tensor data;

	size_t kernel_height, kernel_width;
	size_t out_images;

	void feed_forward(const tensor& in, tensor& out) const
	{
		perform_full_convolution(in, data, out);
	}

	tensor_dims init(tensor_dims input_dims)
	{
		xassert(kernel_width && kernel_height, "Convolution kernels must be non-zero in size");
		xassert(input_dims.width >= kernel_width && input_dims.height >= kernel_height, "Convolution kernels must not exceed the input in dimensions");

		constexpr fp rng_range = (fp)0.1;
		//data.resize_storage({ kernel_height, kernel_width, input_dims.depth * out_images });
		//randomize_range(data, -rng_range, rng_range);
		data = tensor::from_range({-1,-1,-1,0,0,0,1,1,1,-1,0,1,-1,0,1,-1,0,1});
		data.reshape({ 3,3,2 });

		return { input_dims.height - kernel_height + 1, input_dims.width - kernel_width + 1, out_images };
	}

public:
	convolution_layer(size_t height, size_t width, size_t output_images)
		: kernel_height(height), kernel_width(width), out_images(output_images) {}
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
	tensor df(const tensor& observed, const tensor& expected)
	{
		xassert(observed.dims() == expected.dims(), "MSE::df: incompatible-shaped tensor inputs");
		const size_t n = observed.size();

		tensor result(n);
		for (size_t i = 0; i < n; ++i)
			result[i] = 2 * (observed[i] - expected[i]);
		return result;
	}
};





class sgd_optimizer
{
public:
	fp rate;
	size_t max_batch_size = SIZE_MAX;
};






class model
{
	struct _placeholder_loss_function { fp f(const tensor&, const tensor&) { return 0; } };
	struct _placeholder_optimizer {};

	using layer_holder_t = std::variant<dense_layer, convolution_layer, leaky_relu_layer, softmax_layer>;
	using optimizer_holder_t = std::variant<_placeholder_optimizer, sgd_optimizer>;
	using loss_function_holder = std::variant<_placeholder_loss_function, mse_loss_function>;

#define variant_invoke(variant, method, ...) std::visit([&](auto&& stored_obj) { return stored_obj.method(__VA_ARGS__); }, variant)

	struct M
	{
		std::vector<layer_holder_t> layers;
		optimizer_holder_t optimizer;
		loss_function_holder loss_function;
		tensor_dims input_dims;
		tensor_dims output_dims;
	} m;

	friend class model_builder;

	bool has_optimizer() const
	{
		return !std::holds_alternative<_placeholder_optimizer>(m.optimizer);
	}
	bool is_locked() const
	{
		return this->has_optimizer();
	}
	void assert_is_finished(bool is = true) const
	{
		xassert(this->is_locked() == is, "model: action not applicable for {}finished models", this->is_locked() ? "" : "un");
	}

public:
	model(tensor_dims input_layer_dimensions)
		: m{ .input_dims = input_layer_dimensions, .output_dims = input_layer_dimensions } {}

	template<class layer_t>
	void add_layer(layer_t layer)
	{
		assert_is_finished(false);
		m.output_dims = layer.init(m.output_dims);
		m.layers.push_back(std::move(layer));
	}

	template<class optimizer_t>
	void finish(optimizer_t optimizer, loss_function_holder loss_function)
	{
		assert_is_finished(false);
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
	
private:
	fp loss(const tensor& observed, const tensor& expected)
	{
		return variant_invoke(m.loss_function, f, observed, expected);
	}
	fp loss_contribution(const tensor& input, const tensor& expected)
	{
		return loss(predict(input), expected);
	}
};

EXPORT_END
