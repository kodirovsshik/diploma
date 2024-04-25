
module;

#include <variant>
#include <memory>

#include <defs.hpp>

export module diploma.nn;
import diploma.lin_alg;
import diploma.utility;



EXPORT_BEGIN


class leaky_relu_layer
{
	friend class model;

	fp activate(fp x) const
	{
		if (x < 0)
			return negative_side_slope * x;
		else
			return x;
	}

	void feed_forward(tensor& in, tensor& out) const
	{
		for (auto& x : in)
			x = activate(x);
		std::swap(in, out);
	}

	bool check_ok() const
	{
		if (isnan(this->negative_side_slope))
			return false;
		return true;
	}

	tensor_dims init(tensor_dims input_dims)
	{
		return input_dims;
	}

public:
	fp negative_side_slope = fp(NAN);
};


class model
{
	struct _placeholder_optimizer {};
	struct _placeholder_loss_function {};

	using layer_holder_t = std::variant<leaky_relu_layer>;
	using optimizer_holder_t = std::variant<_placeholder_optimizer>;
	using loss_function_holder_t = std::variant<_placeholder_loss_function>;

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
		return !std::holds_alternative<_placeholder_optimizer>(m.optimizer);
	}
	bool is_locked() const
	{
		return this->has_optimizer();
	}
	void assert_is_finished(bool is = true) const
	{
		if (this->is_locked() != is)
			throw;
	}

public:
	model(tensor_dims input_layer_dimensions)
		: m{ .input_dims = input_layer_dimensions, .output_dims = input_layer_dimensions } {}

	template<class layer_t>
	void add_layer(layer_t layer)
	{
		assert_is_finished(false);
		xassert(layer.check_ok(), "Invalid layer configuration for layer {}", m.layers.size());
		m.output_dims = layer.init(m.output_dims);
		m.layers.push_back(std::move(layer));
	}

	template<class optimizer_t, class loss_function_t>
	void finish(optimizer_t optimizer, loss_function_t loss_function)
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
};

EXPORT_END
