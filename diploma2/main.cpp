
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

void main1()
{
	auto img = tensor::from_range({ 0,1,2,3,1,2,3,4,2,3,4,5,3,4,5,6 });
	auto kernel = tensor::from_range({ -1,0,1,-2,0,2,-3,0,3 });
	img.reshape({ 4,4,1 });
	kernel.reshape({ 3,3,1 });

	tensor out;
	perform_full_convolution(img, kernel, out);
}

int main()
{
	//main1(); return 0;

	thread_pool pool;



	cpath dataset_root = R"(C:\dataset_pneumonia\bmp)";
	cpath model_path = dataset_root / "model1.bin";



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
	//auto datagen_func = gen_data_pair_empty;
	auto val_dataset = load_dataset(tag_holder<stub_dataset>{}, "validation", datagen_func, 10);
	auto train_dataset = load_dataset(tag_holder<stub_dataset>{}, "training", datagen_func, 50);
	//auto val_dataset = load_dataset(tag_holder<dataset>{}, "validation", dataset_root / "val");
	//auto train_dataset = load_dataset(tag_holder<dataset>{}, "trainning", dataset_root / "train");



	auto set_default_learning_rate = [&] {
		learning_rate_base = 0.001f;
		learning_rate_decay_rate = 1.1f;
		learning_rate_decay = 0;
	};
	auto create_model = [&] {
		m.set_input_size(input_size(train_dataset));

		m.add_layer(convolution_layer(5, 5, 10));
		m.add_layer(convolution_layer(5, 5, 10));
		m.add_layer(convolution_layer(5, 5, 10));
		m.add_layer(pooling_layer(10, 10));

		m.add_layer(flattening_layer{});
		m.add_layer(dense_layer{ 2 });

		m.finish(mse_loss_function{});

		set_default_learning_rate();
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
	size_t batch_size = max_batch_size;



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

	auto tui_bar = [] { std::println("{:-<49}", ""); };
	
	tui_bar();
	std::println("      |      train         |     validation     |");
	std::println("epoch | loss      | acc    | loss      | acc    |");
	tui_bar();

	using stats_t = model::model_statistics;
	stats_t last_train_stats{}, last_val_stats{};
	fp best_val_loss = INFINITY;

	size_t epochs_passed = 0;
	const size_t epochs_limit = 5;
	const size_t epoch_evaluation_period = 0;

	auto print_stats = [&](stats_t stats) {
		std::print(" {:>10.7f}| {:>7.4f}|", stats.loss, stats.accuracy);
	};

	cursor_pos_holder cursor{ cursor_pos_holder::noacquire };

	const auto xclock = std::chrono::steady_clock::now;
	decltype(xclock() - xclock()) dt_sum{};

	while (true)
	{
		if (epoch_evaluation_period && (epochs_passed % epoch_evaluation_period) == 0)
		{
			cursor.acquire();
			std::print("Evaluating...");
			last_val_stats = m.evaluate(val_dataset, pool);
			cursor.release();

			if (last_val_stats.loss < best_val_loss)
			{
				if (epochs_passed != 0)
					save_model();
				best_val_loss = last_val_stats.loss;
			}
		}

		std::print("{:>6}|", epochs_passed);
		print_stats(last_train_stats);
		print_stats(last_val_stats);
		std::println("");

		if (epochs_passed == epochs_limit) break;

		while (_kbhit()) if (_getch() == ' ') should_enter_menu = true;
		
		if (should_enter_menu) menu();
		if (should_stop) break;

		auto t1 = xclock();
		last_train_stats = m.fit(train_dataset, pool, learning_rate(), batch_size, report_progress);
		auto t2 = xclock();

		dt_sum += t2 - t1;
		++epochs_passed;
	}

	tui_bar();
	std::print("{}s/epoch\n", dt_sum.count() * 1e-9 / epochs_passed);
}
