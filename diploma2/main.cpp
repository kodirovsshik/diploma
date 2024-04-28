
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
	const auto label = tensor::from_range({ x + y, x + 2 * y });
	return { in, label };
}

class stub_dataset
{
	using pair_generator_t = std::function<std::pair<tensor, tensor>()>;

	std::vector<std::pair<tensor, tensor>> dataset;

public:
	stub_dataset(pair_generator_t generator, size_t size)
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
	
	stub_dataset dataset(gen_data_pair, 100);

	model m(dataset.input_size());
	m.add_layer(dense_layer{2});
	m.finish(sgd_optimizer{ .rate = 0.001f }, mse_loss_function{});

	auto x = tensor::from_range({ 1, 2 });
	tensor y;

	while (true)
	{
		y = m.predict(x);
		m.fit(dataset, pool);
	}
	

	//model m(image.dims());
	//m.add_layer(dense_layer{2});
	//m.add_layer(untied_bias_layer{});
	//m.add_layer(softmax_layer{});
	//m.finish(sgd_optimizer{ .rate = 0.001f }, cross_entropy_loss_function{});

	//tensor y;
	//y = m.predict(image);
	//size_t i = 0;
	//while (true)
	//{
	//	m.accumulate_gradient_single(image, label);
	//	y = m.predict(image);
	//	std::print("{}, {}\n", ++i, y[1]);
	//	if (y[1] > 0.99)
	//		__debugbreak();
	//}
}
