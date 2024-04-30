
#include <random>
#include <utility>

#include "defs.hpp"

import diploma.nn;
import diploma.lin_alg;
import diploma.utility;
import diploma.bmp;
import diploma.thread_pool;


std::mt19937_64 rng(rng_seed + 1);

std::pair<tensor, tensor> gen_data_pair()
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

	//cpath dataset_root = "C:/dataset_pn0";
	//dataset_wrapper dataset(dataset_root / "test");
	
	stub_dataset dataset(gen_data_pair, 10);

	model m(dataset.input_size());
	m.add_layer(dense_layer{1});
	m.finish(sgd_optimizer{ .rate = 0.001f }, mse_loss_function{});

	auto x = tensor::from_range({ 1, 2 });
	tensor y;

	auto xclock = std::chrono::steady_clock::now;

	size_t i = 0;
	size_t dt_sum = 0;
	while (true)
	{
		y = m.predict(x);
		std::println("{}, {:.14f}", i++, m.loss(y, tensor::from_range({3})));
		if (i == 100)
			break;

		auto t1 = xclock();
		m.fit(dataset, pool);
		auto t2 = xclock();

		auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
		dt_sum += dt.count();
	}

	std::print("{}ms/epoch", dt_sum / 100);
}
