
module;

#include <math.h>

#include <filesystem>
#include <vector>
#include <span>
#include <fstream>
#include <ranges>
#include <random>

#include "defs.hpp"

export module diploma.nn;

import diploma.serialization;
import diploma.thread_pool;
import diploma.bmp;
import diploma.lin_alg;

namespace fs = std::filesystem;
namespace rn = std::ranges;
namespace vs = rn::views;






struct activation_function_data
{
	using func_ptr = fp(*)(fp);

	func_ptr f, df;
};

fp sigma(fp x)
{
	return 1 / (1 + ::expf(-x));
}
fp sigma_d(fp x)
{
	const fp s = sigma(x);
	return s * (1 - s);
}

activation_function_data sigmoid{ sigma, sigma_d };


constexpr fp leaky_relu_param = (fp)0.01;
activation_function_data leaky_relu{
	lambda(fp x, std::max(x, x * leaky_relu_param)),
	lambda(fp x, (x < 0) ? leaky_relu_param : 1),
};


void activate_vector(fpvector& x, const activation_function_data& activator)
{
	for (auto& val : x)
		val = activator.f(val);
}




export class nn_t
{
private:
	static constexpr uint64_t signature = 0x52EEF7E17876A668;


	void create1(std::span<const unsigned> layer_sizes)
	{
		const size_t n = layer_sizes.size();
		xassert(n >= 2, "Rejected NN topology with {} layers", n + 1);

		this->reset();

		this->weights.resize(n);
		this->biases.resize(n);
		activators.reserve(n);

		for (size_t i = 0; i < n - 1; ++i)
			activators.push_back(leaky_relu);
		activators.push_back(sigmoid);

		for (size_t i = 0; i < n; ++i)
			init_vector(this->biases[i], layer_sizes[i]);

		size_t prev_layer_size = input_neurons;
		for (size_t i = 0; i < n; ++i)
		{
			const size_t next_layer_size = layer_sizes[i];
			init_matrix(this->weights[i], next_layer_size, prev_layer_size);
			prev_layer_size = next_layer_size;
		}
	}


public:
	static constexpr size_t input_neurons = 65536;



	dynarray<matrix> weights;
	dynarray<fpvector> biases;
	dynarray<activation_function_data> activators;

	void create(std::span<const unsigned> layer_sizes)
	{
		nn_t nn;
		nn.create1(layer_sizes);
		std::swap(*this, nn);
	}

	template<class Rng>
	void randomize(Rng&& rng)
	{
		const fp r = 0.1f;
		std::uniform_real_distribution<fp> distr(-r, r);
		for (auto& m : this->weights)
			for (auto& row : m)
				for (auto& w : row)
					w = distr(rng);

		for (auto& v : this->biases)
			for (auto& b : v)
				b = distr(rng);
	}


	bool write(std::ostream& out)
	{
		serializer_t serializer(out);
		serializer(this->signature);

		std::vector<size_t> topology(this->biases.size() + 1);
		topology[0] = this->input_neurons;
		for (size_t i = 0; i < this->biases.size(); ++i)
			topology[i + 1] = this->biases[i].size();

		serializer(topology);
		serializer.write_crc();

		topology = decltype(topology)();

		serializer(this->weights);
		serializer(this->biases);
		serializer.write_crc();

		return (bool)serializer;
	}
	bool write(const std::filesystem::path& p)
	{
		std::ofstream fout(p, std::ios_base::out | std::ios_base::binary);
		return this->write(fout);
	}
	void read(std::istream& in)
	{
		decltype(signature) test_signature{};
		//TODO
		throw;
	}

	void reset()
	{
		this->weights = {};
		this->biases = {};
		this->activators = {};
	}

	static fpvector& get_thread_local_helper_vector()
	{
		static thread_local fpvector v;
		return v;
	}

	void eval(fpvector& x) const
	{
		fpvector& y = get_thread_local_helper_vector();

		const size_t n_layers = this->weights.size();
		for (size_t i = 0; i < n_layers; ++i)
		{
			multiply_mat_vec(this->weights[i], x, y);
			add_vec_vec(this->biases[i], y);
			activate_vector(y, this->activators[i]);
			std::swap(x, y);
		}
	}
};



#define COST_DISTANCE_FUNCTION_DIFFSQ 1
#define COST_DISTANCE_FUNCTION_CROSS_ENTROPY_LOSS 2

#define COST_DISTANCE_FUNCTION COST_DISTANCE_FUNCTION_DIFFSQ


#if COST_DISTANCE_FUNCTION == COST_DISTANCE_FUNCTION_DIFFSQ
fp cost_distance(fp expected, fp observed)
{
	fp d = observed - expected;
	return d * d;
}
fp cost_distance_dfdo(fp expected, fp observed)
{
	fp d = observed - expected;
	return 2 * d;
}
#elif COST_DISTANCE_FUNCTION == COST_DISTANCE_FUNCTION_CROSS_ENTROPY_LOSS
fp cost_distance(fp expected, fp observed)
{
	return -expected * log(observed);
}
fp cost_distance_dfdo(fp expected, fp observed)
{
	return -expected / observed;
}
#endif

struct data_pair
{
	fpvector input;
	fpvector output;
};

export fp nn_eval_cost(const nn_t& nn, const std::vector<data_pair>& dataset, thread_pool& pool)
{
	struct alignas(64) thread_state
	{
		size_t n = 0;
		fp total_loss = 0;
	};

	std::vector<thread_state> states(pool.size());

	auto worker = [&]
	(const size_t i, const size_t thread_id)
	{
		auto& state = states[thread_id];
		const auto& data = dataset[i];

		fpvector eval_result = data.input;
		nn.eval(eval_result);
		xassert(eval_result.size() == data.output.size(), "Incompatible sized vectors");
		for (size_t i = 0; i < eval_result.size(); ++i)
			state.total_loss += cost_distance(data.output[i], eval_result[i]);
		state.n++;
	};

	pool.schedule_sized_work(0, dataset.size(), worker);
	pool.barrier();

	fp sum = 0;
	size_t div = 0;
	for (const auto& x : states)
	{
		sum += x.total_loss;
		div += x.n;
	}

	return sum / div;
}



template<class T>
concept indexable = requires(const T & x)
{
	x[size_t{}];
};


template<class Resizeable, class Sizeable>
void resize_to_match(Resizeable& r, const Sizeable& s)
{
	size_t n = 0;

	static constexpr bool can_size = requires() { {s.size()} -> std::convertible_to<size_t>; };
	static constexpr bool can_resize = requires(size_t n) { {r.resize(n)}; };

	if constexpr (can_size)
	{
		n = s.size();
		if constexpr (can_resize)
			r.resize(n);
	}

	if constexpr (indexable<Resizeable> && indexable<Sizeable>)
		for (size_t i = 0; i < n; ++i)
			resize_to_match(r[i], s[i]);
}

auto nn_eval_cost_gradient(const nn_t& nn, const data_pair& pair)
{
	auto& data = pair.input;
	auto& expected = pair.output;

	auto layer_size = lambdac(size_t i, nn.biases[i].size());
	const size_t layers_count = nn.biases.size();

	xassert(expected.size() == layer_size(layers_count - 1), "Wrong expected output size");

	static thread_local dynarray<matrix> dweights;
	static thread_local dynarray<fpvector> dbiases;

	static thread_local dynarray<fpvector> sums;
	static thread_local dynarray<fpvector> activations;

	static thread_local fpvector dactivations, new_dactivations;

	resize_to_match(dweights, nn.weights);
	resize_to_match(dbiases, nn.biases);

	activations.resize(layers_count);
	sums.resize(layers_count);

	{
		const fpvector* prev_layer = &data;

		for (size_t i = 0; i < layers_count; ++i)
		{
			multiply_mat_vec(nn.weights[i], *prev_layer, sums[i]);
			add_vec_vec(nn.biases[i], sums[i]);
			resize_to_match(activations[i], sums[i]);
			for (size_t j = 0; j < sums[i].size(); ++j)
				activations[i][j] = nn.activators[i].f(sums[i][j]);
			prev_layer = &activations[i];
		}
	}

	{
		auto get_size = lambda(const auto & x, x.size());
		auto sizes_range = vs::transform(nn.biases, get_size);
		const size_t max_layer_size = rn::max(sizes_range);

		dactivations.reserve(max_layer_size);
		new_dactivations.reserve(max_layer_size);
	}

	dactivations.resize(layer_size(layers_count - 1));
	for (size_t i = 0; i < dactivations.size(); ++i)
		dactivations[i] = cost_distance_dfdo(expected[i], activations.back()[i]);

	for (size_t k = layers_count - 1; true; --k)
	{
		const size_t layer_idx = k;
		const size_t current_layer_size = layer_size(layer_idx);
		const size_t prev_layer_size = layer_idx == 0 ? nn.input_neurons : layer_size(layer_idx - 1);
		
		for (size_t i = 0; i < current_layer_size; ++i)
			dbiases[layer_idx][i] = dactivations[i] * nn.activators[layer_idx].df(sums[layer_idx][i]);

		for (size_t i = 0; i < current_layer_size; ++i)
			for (size_t j = 0; j < prev_layer_size; ++j)
				dweights[layer_idx][i][j] = dbiases[layer_idx][i] * (layer_idx == 0 ? data[j] : activations[layer_idx - 1][j]);
		
		if (k == 0)
			break;

		new_dactivations.clear();
		new_dactivations.resize(prev_layer_size);
		for (size_t j = 0; j < new_dactivations.size(); ++j)
			for (size_t i = 0; i < current_layer_size; ++i)
				new_dactivations[j] += dbiases[layer_idx][i] * nn.weights[layer_idx][i][j];

		std::swap(dactivations, new_dactivations);
	}

	fp cost = 0;
	for (size_t i = 0; i < expected.size(); ++i)
		cost += cost_distance(expected[i], activations.back()[i]);

	return std::tuple<const decltype(dweights)&, const decltype(dbiases)&, fp>{ dweights, dbiases, cost };
}

export fp nn_apply_gradient_descend_iteration(nn_t& nn, const dynarray<data_pair>& dataset, size_t iteration, thread_pool& pool, fp rate)
{
	struct alignas(64) thread_state
	{
		dynarray<matrix> dweights;
		dynarray<fpvector> dbiases;
		fp cost = 0;
		size_t n = 0;
	};

	static constexpr bool stochastic = true;

	const size_t dataset_split_count = stochastic ? 10 : 1;
	iteration %= dataset_split_count;
	const size_t dataset_split_size = dataset.size() / dataset_split_count;
	const size_t dataset_begin = dataset_split_count * iteration;

	dynarray<thread_state> states;

	auto worker = [&]
	(size_t i, size_t thread_id)
	{
		auto& state = states[thread_id];
		const auto& [dweights, dbiases, cost] = nn_eval_cost_gradient(nn, dataset[i]);

		if (state.dweights.size() == 0)
		{
			state.dweights = dweights;
			state.dbiases = dbiases;
		}
		else
		{
			for (size_t i = 0; i < dweights.size(); ++i)
				add_mat_mat(dweights[i], state.dweights[i]);
			for (size_t i = 0; i < dbiases.size(); ++i)
				add_vec_vec(dbiases[i], state.dbiases[i]);
		}
		state.cost += cost;
		state.n++;
	};

	resize_to_match(states, pool);
	pool.schedule_sized_work(dataset_begin, dataset_split_size, worker);
	pool.barrier();


	thread_state result = std::move(states[0]);

	for (size_t i = 1; i < states.size(); ++i)
	{
		result.cost += states[i].cost;
		result.n += states[i].n;

		for (size_t j = 0; j < result.dweights.size(); ++j)
			add_mat_mat(states[i].dweights[j], result.dweights[j]);
		for (size_t j = 0; j < result.dbiases.size(); ++j)
			add_vec_vec(states[i].dbiases[j], result.dbiases[j]);
	}

	result.cost /= result.n;
	
	size_t params_count = 0;
	for (size_t i = 0; i < result.dbiases.size(); ++i)
		params_count += result.dbiases[i].size();
	for (size_t i = 0; i < result.dweights.size(); ++i)
	{
		const size_t n = result.dweights[i].size();
		params_count += n * n;
	}

	const fp scale = -rate / (result.n);
	//const fp scale = -rate / (result.n * params_count);

	for (size_t l = 0; l < result.dweights.size(); ++l)
	{
		auto& weights = result.dweights[l];
		for (size_t i = 0; i < weights.size(); ++i)
			for (size_t j = 0; j < weights[i].size(); ++j)
				weights[i][j] *= scale;
	}
	for (size_t l = 0; l < result.dbiases.size(); ++l)
	{
		auto& biases = result.dbiases[l];
		for (size_t i = 0; i < biases.size(); ++i)
			biases[i] *= scale;
	}

	for (size_t l = 0; l < result.dweights.size(); ++l)
		add_mat_mat(result.dweights[l], nn.weights[l]);
	for (size_t l = 0; l < result.dbiases.size(); ++l)
		add_vec_vec(result.dbiases[l], nn.biases[l]);

	return result.cost;
}





using cpath = const std::filesystem::path&;

template<class Img>
auto read_dataset(const bool grayscale, cpath class_positive, cpath class_negative)
{
	std::vector<data_pair> dataset;

	Img img;

	auto traverse = [&]
	(cpath p, const fpvector& output)
		{
			for (auto& entry : std::filesystem::directory_iterator(p))
			{
				img.read(entry.path(), grayscale);
				dataset.emplace_back(std::move(img.planes[0]), output);
			}
		};

	traverse(class_positive, { 1, 0 });
	traverse(class_negative, { 0, 1 });

	return dataset;
}
export auto read_main_dataset()
{
	using T = bmp_image<fp>;
	return read_dataset<T>(true, "C:\\dataset\\training\\Positiv1000", "C:\\dataset\\training\\Negativ1000");
}





export auto create_preset_topology_nn()
{
	const auto layers = { 40u, 10000u, 2u };
	nn_t nn;
	nn.create(layers);
	return nn;
}
