
import diploma.nn;
import diploma.lin_alg;
import diploma.utility;
import diploma.image;
import diploma.thread_pool;
import diploma.dataset;
import diploma.data_generator;

import <random>;
import <utility>;

import "defs.hpp";



template<class tag>
struct tag_holder {};

int main()
{

	const auto xclock = std::chrono::steady_clock::now;
	decltype(xclock() - xclock()) dt_sum{};



	thread_pool pool;



	cpath dataset_root = R"(C:\dataset_pneumonia\bmp)";
	cpath model_path = dataset_root / "model.bin";



	fp learning_rate_base;
	fp learning_rate_decay_rate;
	int learning_rate_decay;

	auto learning_rate = [&] { return learning_rate_base * powf(learning_rate_decay_rate, -(fp)learning_rate_decay); };



	static constexpr uint64_t serialization_magic = 0xC5DA6D48C547CDEC;

	model m;
	auto try_load_model = [&] {
		std::ifstream fin(model_path);
		deserializer_t deserializer(fin);

		uint64_t test_magic{};
		deserializer(test_magic);
		if (test_magic != serialization_magic)
			return false;

		deserializer(learning_rate_base);
		deserializer(learning_rate_decay_rate);
		deserializer(learning_rate_decay);
		deserializer.test_crc();

		return m.deserialize(deserializer) && deserializer.test_crc() && deserializer;
	};
	auto save_model = [&] {
		std::ofstream fout(model_path);
		serializer_t serializer(fout);

		serializer(serialization_magic);
		serializer(learning_rate_base);
		serializer(learning_rate_decay_rate);
		serializer(learning_rate_decay);
		serializer.write_crc();
		m.serialize(serializer);
	};



	std::println("Dataset root: {}", (char*)dataset_root.generic_u8string().data());
	std::print("Trying to load existing model... ");
	if (try_load_model())
	{
		std::println("success");
		//TODO: enter online classification mode prompt
	}
	else
	{
		std::println("failure");
		std::println("New model will be created");
	}



	auto datagen_func = gen_data_pair_circle_square;

	auto load_dataset = [&]<class DatasetType>(tag_holder<DatasetType>, const char* type, auto&& ...args) {
		std::print("Loading {} dataset... ", type);
		auto result = DatasetType(std::forward<decltype(args)>(args)...);
		std::println("{} items loaded", size(result));
		return result;
	};

	auto val_dataset = load_dataset(tag_holder<dataset>{}, "validation", dataset_root / "val");
	auto train_dataset = load_dataset(tag_holder<dataset>{}, "trainning", dataset_root / "train");



	auto create_model = [&] {
		m.set_input_size(input_size(train_dataset));

		m.add_layer(convolution_layer(3, 3, 16));
		m.add_layer(pooling_layer{ 2, 2 });
		m.add_layer(tied_bias_layer{});
		m.add_layer(leaky_relu_layer{});

		m.add_layer(convolution_layer(3, 3, 16));
		m.add_layer(pooling_layer{ 2, 2 });
		m.add_layer(tied_bias_layer{});
		m.add_layer(leaky_relu_layer{});

		m.add_layer(convolution_layer(3, 3, 16));
		m.add_layer(pooling_layer{ 2, 2 });
		m.add_layer(tied_bias_layer{});
		m.add_layer(leaky_relu_layer{});

		m.add_layer(flattening_layer{});

		m.add_layer(dense_layer{ 32 });
		m.add_layer(untied_bias_layer{});
		m.add_layer(leaky_relu_layer{});

		m.add_layer(dense_layer{ 32 });
		m.add_layer(untied_bias_layer{});
		m.add_layer(leaky_relu_layer{});

		m.add_layer(dense_layer{ 2 });
		m.add_layer(untied_bias_layer{});
		m.add_layer(softmax_layer{});

		//m.finish(mse_loss_function{});
		m.finish(cross_entropy_loss_function{});

		learning_rate_base = 0.001f;
		learning_rate_decay_rate = 1.1f;
		learning_rate_decay = 0;
	};
	if (m.get_layer_count() == 0)
	{
		std::println("Creating new model");
		create_model();
		std::println("All parameters set to random values");
		std::println("All hyperparameters set to default values");
	}


	const size_t max_batch_size = size(train_dataset);
	const size_t small_batch_size = std::min<size_t>(200, max_batch_size / 3);
	size_t batch_size = max_batch_size;



	std::println("Batch size: {}", batch_size);
	std::println("Learning rate: {}", learning_rate());



	bool should_exit = false;
	auto menu = [&] {
		std::print("Menu:\n learning rate control: + -\n batch size control: * /\n");

		auto mutate_var_ = [&](const char* name, auto& val, auto new_val)
			{
				std::print("{}: {} -> ", name, val);
				std::print("{}\n", val = new_val);
			};
#define mutate_var(var, new_val) mutate_var_(#var, var, new_val)

		auto mutate_nearning_rate = [&](int decay_delta) {
			std::print("learning_rate: {} -> ", learning_rate());
			learning_rate_decay -= decay_delta;
			std::println("{}", learning_rate());
			};

		bool exit_menu = false;
		while (!exit_menu) switch (_getch())
		{
		case 27: mutate_var(should_exit, !should_exit); break;
		case '*': mutate_var(batch_size, max_batch_size); break;
		case '/': mutate_var(batch_size, small_batch_size); break;

		case '+': mutate_nearning_rate(+1); break;
		case '-': mutate_nearning_rate(-1); break;

		case ' ': exit_menu = true; break;
		//case PAUSE: if (DebuggerAttached()) __debugbreak();
		default: break;
		}
		};



	std::println("Training is running on {} thread{}", pool.size(), pool.size() == 1 ? "" : "s");
	std::println("epochs_passed, train_loss, train_acc, val_loss, val_acc:");

	fp last_train_acc = 0, last_train_loss = 0;
	fp top_val_acc = 0;

	size_t epochs_passed = 0;
	const size_t epochs_limit = -1;
	save_model();

	while (true)
	{
		if ((epochs_passed % 1) == 0)
		{
			auto stats = m.evaluate(val_dataset, pool);
			std::println("{:>6}, {:<7.4f}, {:<7.4f}, {:<7.4f}, {:<7.4f}", epochs_passed, last_train_loss, last_train_acc, stats.loss, stats.accuracy);
			if (stats.accuracy > top_val_acc)
			{
				if (epochs_passed != 0)
					save_model();
				top_val_acc = stats.accuracy;
			}
		}

		while (_kbhit()) if (_getch() == ' ') menu();
		if (should_exit) break;

		auto t1 = xclock();
		auto stats = m.fit(train_dataset, pool, learning_rate(), batch_size);
		auto t2 = xclock();

		last_train_acc = stats.accuracy;
		last_train_loss = stats.loss;

		dt_sum += t2 - t1;
		++epochs_passed;
		if (epochs_passed == epochs_limit) break;
	}

	std::print("{}s/epoch\n", dt_sum.count() * 1e-9 / epochs_passed);
}
