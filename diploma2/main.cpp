
#include <random>
#include <utility>

#include "defs.hpp"

import diploma.nn;
import diploma.lin_alg;
import diploma.utility;
import diploma.bmp;
import diploma.thread_pool;


std::mt19937_64 rng(rng_seed + 1);

std::pair<tensor, tensor> gen_data_pair_max1()
{
	static thread_local std::uniform_real_distribution<fp> dist(-10, 10);

	const fp x = dist(rng);

	const auto in = tensor::from_range({ x });
	const auto label = tensor::from_range({ std::max<fp>(x, 1) });
	return { in, label };
}
std::pair<tensor, tensor> gen_data_pair_sum()
{
	static thread_local std::uniform_real_distribution<fp> dist(0, 10);

	const fp x = dist(rng);
	const fp y = dist(rng);

	const auto in = tensor::from_range({ x, y });
	const auto label = tensor::from_range({ x + y });
	return { in, label };
}

class stub_dataset
{
	std::vector<std::pair<tensor, tensor>> dataset;

public:
	template<class pair_generator_t>
	stub_dataset(pair_generator_t&& generator, size_t size)
	{
		dataset.reserve(size);
		while (size --> 0)
			dataset.emplace_back(generator());
	}

	void shuffle() noexcept {}

	const auto& operator[](size_t i) const
	{
		return dataset[i];
	}

	tensor_dims input_size() const
	{
		return dataset[0].first.dims();
	}

	size_t size() const noexcept
	{
		return dataset.size();
	}
};



int main()
{
	thread_pool pool;
	
	stub_dataset train_dataset(gen_data_pair_max1, 1000);
	stub_dataset val_dataset(gen_data_pair_max1, 10);

	model m(train_dataset.input_size());
	m.add_layer(convolution_layer(1, 1, 2));
	m.add_layer(tied_bias_layer{});
	m.add_layer(flattening_layer{});
	m.add_layer(pooling_layer{2, 1});
	m.finish(sgd_optimizer(0.001f), mse_loss_function{});

	const auto xclock = std::chrono::steady_clock::now;

	size_t i = 0;
	decltype(xclock() - xclock()) dt_sum{};
	while (true)
	{
		if ((i % 200) == 0)
			std::println("{}, {}", i, m.evaluate(val_dataset, pool));
		if (i == 35000 && false)
			break;

		auto t1 = xclock();
		m.fit(train_dataset, pool);
		auto t2 = xclock();

		dt_sum += t2 - t1;
		++i;
	}

	std::print("{}/epoch\n{} -> {}\n{} -> {}", dt_sum / (__int64)i, 
		2, m.predict(tensor::from_range({ 2 }))[0],
		-2, m.predict(tensor::from_range({ -2 }))[0]
	);
}
