
#include <random>
#include <utility>

#include "defs.hpp"

import diploma.nn;
import diploma.lin_alg;
import diploma.utility;
import diploma.image;
import diploma.thread_pool;
import diploma.dataset;
import diploma.data_generator;



int main()
{
	datagen_rng.discard(65536);


	//cpath p = "D:/a.qoi";

	//image img;
	//auto& image = img.data;

	//while (true)
	//{
	//	//image = gen_circle_image_tensor();
	//	image = gen_square_image_tensor();
	//	img.write_qoi(p, { .colors = threestate::yes });
	//	system(("start " + p.generic_string()).data());
	//	_getch();
	//}

	thread_pool pool;
	
	auto datagen_func = gen_data_pair_circle_square;
	stub_dataset train_dataset(datagen_func, 150);
	stub_dataset val_dataset(datagen_func, 50);

	model m(input_size(train_dataset));

	m.add_layer(convolution_layer(5, 5, 3));
	m.add_layer(pooling_layer{2, 2});
	m.add_layer(tied_bias_layer{});
	m.add_layer(leaky_relu_layer{});

	m.add_layer(convolution_layer(5, 5, 5));
	m.add_layer(pooling_layer{2, 2});
	m.add_layer(tied_bias_layer{});
	m.add_layer(leaky_relu_layer{});

	//m.add_layer(convolution_layer(5, 5, 30));
	//m.add_layer(pooling_layer{2, 2});
	//m.add_layer(tied_bias_layer{});
	//m.add_layer(leaky_relu_layer{});

	m.add_layer(flattening_layer{});

	//m.add_layer(dense_layer{16});
	//m.add_layer(leaky_relu_layer{});

	m.add_layer(dense_layer{4});
	m.add_layer(leaky_relu_layer{});

	m.add_layer(dense_layer{2});
	m.add_layer(softmax_layer{});

	m.finish(cross_entropy_loss_function{});

	const auto xclock = std::chrono::steady_clock::now;

	const size_t max_batch_size = size(train_dataset);
	const size_t small_batch_size = 20;
	size_t i = 0, batch_size = max_batch_size;
	decltype(xclock() - xclock()) dt_sum{};

	fp last_train_acc = 0, last_train_loss = 0;

	const fp learning_rate_decay_rate = 1.1f;
	int learning_rate_decay = 0;
	const fp learning_rate_base = 0.008f;
	auto learning_rate = [&] { return learning_rate_base * powf(learning_rate_decay_rate, -(fp)learning_rate_decay); };

	std::println("epoch, train_loss, train_acc, val_loss, val_acc");
	while (true)
	{
		if ((i % 1) == 0)
		{
			auto stats = m.evaluate(val_dataset, pool);
			std::println("{:6}, {:<10}, {:<10}, {:<10}, {:<10}", i, last_train_loss, last_train_acc, stats.loss, stats.accuracy);
		}

		bool exit = false;

		if (_kbhit()) switch (_getch())
		{
		case 27: exit = true; break;
		case '*': batch_size = max_batch_size; break;
		case '/': batch_size = small_batch_size; break;

		case '+':
			std::print("learning_rate: {} -> ", learning_rate());
			learning_rate_decay--;
			std::print("{}\n", learning_rate());
			break;

		case '-':
			std::print("learning_rate: {} -> ", learning_rate());
			learning_rate_decay++;
			std::print("{}\n", learning_rate());
			break;

		case ' ': __debugbreak();
		default: break;
		}
		if (exit) break;

		auto t1 = xclock();
		auto stats = m.fit(train_dataset, pool, learning_rate(), batch_size);
		auto t2 = xclock();

		last_train_acc = stats.accuracy;
		last_train_loss = stats.loss;

		dt_sum += t2 - t1;
		++i;
	}

	//std::print("{}/epoch\n{} -> {}\n{} -> {}", dt_sum / (__int64)i, 
	//	2, m.predict(tensor::from_range({ 2 }))[0],
	//	-2, m.predict(tensor::from_range({ -2 }))[0]
	//);
}
