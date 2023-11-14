
module;

#include <math.h>

#include <filesystem>
#include <vector>
#include <span>
#include <fstream>

#include "defs.hpp"

export module diploma.nn;

import diploma.serialization;
import diploma.thread_pool;
import diploma.bmp;

namespace fs = std::filesystem;





export template<class T>
using dynarray = std::vector<T>;

export using fp = float;
using fpvector = dynarray<fp>;
using matrix = dynarray<fpvector>;


void multiply_mat_vec(const matrix& m, const fpvector& v, fpvector& x)
{
	if (m.size() == 0)
	{
		x.clear();
		return;
	}

	const size_t n1 = m.size();
	const size_t n2 = m[0].size();
	xassert(n2 == v.size(), "incompatible multiplication: {}x{} by {}", n1, n2, v.size());

	x.resize(n1);
	for (size_t i = 0; i < n1; ++i)
	{
		x[i] = 0;
		for (size_t j = 0; j < n2; ++j)
			x[i] += m[i][j] * v[j];
	}
}
void add_vec_vec(const fpvector& x, fpvector& y)
{
	const size_t n1 = x.size();
	const size_t n2 = y.size();
	xassert(n1 == n2, "incompatible addition: {} and {}", n1, n2);

	for (size_t i = 0; i < n1; ++i)
		y[i] += x[i];
}



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


public:
	static constexpr size_t input_neurons = 65536;


	static void init_vector(fpvector& arr, size_t n)
	{
		arr.clear();
		arr.resize(n);
	}

	static void init_matrix(matrix& mat, size_t rows, size_t columns)
	{
		mat.resize(rows);
		for (auto& row : mat)
			init_vector(row, columns);
	}

	dynarray<matrix> weights;
	dynarray<fpvector> biases;
	dynarray<activation_function_data> activators;

	void create(std::span<const unsigned> layer_sizes)
	{
		const size_t n = layer_sizes.size();
		xassert(n >= 2, "Rejected NN topology with {} layers", n + 1);

		this->reset();

		weights.resize(n);
		biases.resize(n);
		activators.reserve(n);

		for (size_t i = 0; i < n - 1; ++i)
			activators.push_back(leaky_relu);
		activators.push_back(sigmoid);

		for (size_t i = 0; i < n; ++i)
			init_vector(biases[i], layer_sizes[i]);

		size_t prev_layer_size = input_neurons;
		for (size_t i = 0; i < n; ++i)
		{
			const size_t next_layer_size = layer_sizes[i];
			init_matrix(weights[i], next_layer_size, prev_layer_size);
			prev_layer_size = next_layer_size;
		}
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
	void write(const std::filesystem::path& p)
	{
		std::ofstream fout(p, std::ios_base::out | std::ios_base::binary);
		const bool ok = this->write(fout);
		if (!ok)
			std::print("Failed to serialize to {}\n", p.string());
	}
	void read(std::istream& in)
	{
		decltype(signature) test_signature{};
		//TODO
		throw;
	}

	void reset()
	{
		this->weights.clear();
		this->biases.clear();
		this->activators.clear();
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

fp cross_entropy_loss(const fpvector& ex, const fpvector& obs)
{
	using std::log;
	xassert(ex.size() == obs.size(), "Incompatible vectors: {} and {}", ex.size(), obs.size());
	fp sum = 0;
	for (size_t i = 0; i < ex.size(); ++i)
	{
#ifdef _DEBUG
		fp _ = log(obs[i]);
		xassert(_ == _, "Unexpected NAN detected");
#endif
		sum -= ex[i] * log(obs[i]);
	}
	return sum;
}

struct data_pair
{
	fpvector input;
	fpvector output;
};

fp nn_eval_over_dataset(const std::vector<data_pair>& dataset, thread_pool& pool)
{
	struct alignas(64) thread_state
	{
		size_t n = 0;
		fp total_loss = 0;
	};

	std::vector<thread_state> states(pool.size());

	return 0;
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
