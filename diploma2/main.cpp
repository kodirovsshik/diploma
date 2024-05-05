
import diploma.nn;
import diploma.lin_alg;
import diploma.utility;
import diploma.image;
import diploma.thread_pool;
import diploma.dataset;
import diploma.data_generator;
import diploma.conio;

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
		std::ifstream fin(model_path, std::ios::in | std::ios::binary);
		deserializer_t deserializer(fin);

		uint64_t test_magic{};
		deserializer(test_magic);
		if (test_magic != serialization_magic)
			return false;

		deserializer(learning_rate_base);
		deserializer(learning_rate_decay_rate);
		deserializer(learning_rate_decay);
		if (!deserializer.test_crc())
			return false;

		return deserializer(m) && deserializer.test_crc();
	};
	auto save_model = [&] {
		std::ofstream fout(model_path, std::ios::out | std::ios::binary);
		serializer_t serializer(fout);

		serializer(serialization_magic);
		serializer(learning_rate_base);
		serializer(learning_rate_decay_rate);
		serializer(learning_rate_decay);
		serializer.write_crc();
		serializer(m);
		serializer.write_crc();
		xassert(serializer, "Failed to save model");
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



	auto load_dataset = [&]<class DatasetType>(tag_holder<DatasetType>, const char* type, auto&& ...args) {
		std::print("Loading {} dataset... ", type);
		auto result = DatasetType(std::forward<decltype(args)>(args)...);
		std::println("{} items loaded", size(result));
		return result;
	};

	auto datagen_func = gen_data_pair_circle_square;
	//auto val_dataset = load_dataset(tag_holder<stub_dataset>{}, "validation", datagen_func, 20);
	//auto train_dataset = load_dataset(tag_holder<stub_dataset>{}, "training", datagen_func, 500);
	auto val_dataset = load_dataset(tag_holder<dataset>{}, "validation", dataset_root / "val");
	auto train_dataset = load_dataset(tag_holder<dataset>{}, "trainning", dataset_root / "train");



	auto create_model = [&] {
		m.set_input_size(input_size(train_dataset));

		m.add_layer(convolution_layer(5, 5, 10));
		m.add_layer(pooling_layer{ 2, 2 });
		m.add_layer(tied_bias_layer{});
		m.add_layer(leaky_relu_layer{});

		m.add_layer(convolution_layer(5, 5, 20));
		m.add_layer(pooling_layer{ 2, 2 });
		m.add_layer(tied_bias_layer{});
		m.add_layer(leaky_relu_layer{});

		m.add_layer(convolution_layer(5, 5, 20));
		m.add_layer(pooling_layer{ 2, 2 });
		m.add_layer(tied_bias_layer{});
		m.add_layer(leaky_relu_layer{});

		m.add_layer(convolution_layer(5, 5, 20));
		m.add_layer(pooling_layer{ 2, 2 });
		m.add_layer(tied_bias_layer{});
		m.add_layer(leaky_relu_layer{});

		m.add_layer(flattening_layer{});

		m.add_layer(dense_layer{ 32 });
		m.add_layer(untied_bias_layer{});
		m.add_layer(leaky_relu_layer{});

		m.add_layer(dense_layer{ 16 });
		m.add_layer(untied_bias_layer{});
		m.add_layer(leaky_relu_layer{});

		m.add_layer(dense_layer{ 2 });
		m.add_layer(untied_bias_layer{});
		m.add_layer(softmax_layer{});

		m.finish(mse_loss_function{});
		//m.finish(cross_entropy_loss_function{});

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
	const size_t small_batch_size = std::min<size_t>(500, max_batch_size / 3);
	size_t batch_size = small_batch_size;



	std::println("Batch size: {}", batch_size);
	std::println("Learning rate: {}", learning_rate());



	bool report_progress = true;



	bool should_stop = false;
	bool should_enter_menu = false;

	auto menu = [&] {
		cursor_pos_holder cursor;
		size_t lines = 0;

		std::println("Menu:"); ++lines;
		std::println(" learning rate control: + -");  ++lines;
		std::println(" batch size control: * / w s");  ++lines;
		std::println(" live fitting display control: .");  ++lines;
		std::println(" exit menu: (space)");  ++lines;
		std::println(" exit menu for 1 iteration: `");  ++lines;

		should_enter_menu = false;

		auto mutate_var_ = [&](const char* name, auto& val, auto new_val)
			{
				std::print(" {}: {} -> ", name, val);
				std::print("{}\n", val = new_val);
			};
#define mutate_var(var, new_val) mutate_var_(#var, var, new_val)

		auto mutate_learning_rate = [&](int decay_delta) {
			std::print(" learning_rate: {} -> ", learning_rate());
			learning_rate_decay -= decay_delta;
			std::println("{}", learning_rate());
			};

		bool exit_menu = false;
		while (!exit_menu)
		{
			cursor_pos_holder _;

			const auto ch = _getch();
			clear_line();

			switch (ch)
			{
			case 27: mutate_var(should_stop, !should_stop); break;
			case 'w': mutate_var(batch_size, std::min<size_t>(max_batch_size, batch_size + 10)); break;
			case 's': mutate_var(batch_size, std::max<size_t>(batch_size, 10) - 10); break;
			case '*': mutate_var(batch_size, max_batch_size); break;
			case '/': mutate_var(batch_size, small_batch_size); break;
			case '.': mutate_var(report_progress, !report_progress); break;

			case '+': mutate_learning_rate(+1); break;
			case '-': mutate_learning_rate(-1); break;

			case '`': should_enter_menu = true; [[fallthrough]];
			case ' ': exit_menu = true; break;
			//case PAUSE: if (DebuggerAttached()) __debugbreak();
			default: break;
			}
		}

		cursor.restore();
		for (size_t i = 0; i < lines; ++i)
		{
			clear_line(); std::println("");
		}
	};



	std::println("Training is running on {} thread{}", pool.size(), pool.size() == 1 ? "" : "s");
	std::println("epochs_passed, train_loss, train_acc, val_loss, val_acc:");

	fp last_train_acc = 0, last_train_loss = 0;
	fp best_val_loss = INFINITY;

	size_t epochs_passed = 0;
	const size_t epochs_limit = -1;
	const size_t epoch_evaluation_period = 1;

	cursor_pos_holder cursor{ cursor_pos_holder::noacquire };

	while (true)
	{
		if ((epochs_passed % epoch_evaluation_period) == 0)
		{
			cursor.acquire();
			std::print("Evaluating...");
			auto val_stats = m.evaluate(val_dataset, pool);
			cursor.release();

			std::println("{:>6}, {:<10.7f}, {:<7.4f}, {:<10.7f}, {:<7.4f}", epochs_passed, last_train_loss, last_train_acc, val_stats.loss, val_stats.accuracy);
			if (val_stats.loss < best_val_loss)
			{
				if (epochs_passed != 0)
					save_model();
				best_val_loss = val_stats.loss;
			}
		}

		if (epochs_passed == epochs_limit) break;

		while (_kbhit()) if (_getch() == ' ') should_enter_menu = true;
		
		if (should_enter_menu) menu();
		if (should_stop) break;

		auto t1 = xclock();
		auto train_stats = m.fit(train_dataset, pool, learning_rate(), batch_size, report_progress);
		auto t2 = xclock();

		last_train_acc = train_stats.accuracy;
		last_train_loss = train_stats.loss;

		dt_sum += t2 - t1;
		++epochs_passed;
	}

	std::print("{}s/epoch\n", dt_sum.count() * 1e-9 / epochs_passed);
}
