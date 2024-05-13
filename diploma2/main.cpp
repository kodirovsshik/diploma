
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
	const size_t Ni = 10, Nk = 5;
	tensor img(Ni, Ni), kernel(Nk, Nk);

	for (size_t y = 0; y < Ni; ++y)
		for (size_t x = 0; x < Ni; ++x)
			img(y, x) = fp(y + x);
	for (size_t y = 0; y < Nk; ++y)
		for (size_t x = 0; x < Nk; ++x)
			kernel(y, x) = (y + 1) * ((fp)2 * x / (Nk - 1) - 1);

	tensor out;
	perform_full_convolution<true>(img, kernel, out);
}


thread_pool pool(-1);


cpath dataset_root = R"(C:\dataset_pneumonia\bmp)";
cs model_codename = "test_cs3t5d16d8d2_ce";
cpath model_path = dataset_root / (model_codename + ".bin");
cpath stats_path = dataset_root / (model_codename + ".stats");

model m;


int main()
{
	//main1(); return 0;

	fp learning_rate_base;
	fp learning_rate_decay_rate;
	int learning_rate_decay;

	auto learning_rate = [&] { return learning_rate_base * powf(learning_rate_decay_rate, -(fp)learning_rate_decay); };



	using stats_t = model::model_statistics;

	struct learn_statistics
	{
		stats_t train, val;
	};

	std::vector<learn_statistics> stats_history;
	learn_statistics last_stats{};
	bool last_val_up_to_date = false;



	auto print_last_validation_stats = [&] {
		std::println("Current validation loss: {:.10f}", last_stats.val.loss);
		std::println("Current validation accuracy: {:.5f}", last_stats.val.accuracy); 
	};
	auto save_stats = [&] (cpath p = stats_path) __declspec(noinline)
	{
		std::ofstream fout(p);
		if (!fout.is_open())
		{
			std::println("Failed to save stats to {}", (char*)p.generic_u8string().data());
			return;
		}

		auto save_stat = [&](size_t i, const auto& stats) { std::println(fout, "{}, {:.10f}, {:.5f}", i, stats.loss, stats.accuracy); };

		auto write_stats_type = [&](const char* msg, stats_t learn_statistics::*member) {
			std::println(fout, "{}:", msg);

			for (size_t i = 0; i < stats_history.size(); ++i)
				save_stat(i, stats_history[i].*member);
		};

		write_stats_type("training", &learn_statistics::train);
		write_stats_type("validation", &learn_statistics::val);
		if (last_val_up_to_date)
			save_stat(stats_history.size(), last_stats.val);
		std::println("Stats saved to {}", (char*)p.generic_u8string().data());
	};



	static constexpr uint64_t serialization_magic = 0xC5DA6D48C547CDEC;

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

		deserializer(m);
		deserializer(stats_history);
		deserializer(last_stats.val);
		if (!deserializer.test_crc())
			return false;

		return (bool)deserializer;
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
		serializer(stats_history);
		serializer(last_stats.val);
		serializer.write_crc();
		xassert(serializer, "Failed to save model");
	};



	size_t current_step = 0;



	std::println("Dataset root: {}", (char*)dataset_root.generic_u8string().data());
	std::print("Trying to load model {} ... ", model_codename);
	if (try_load_model())
	{
		std::println("success");
		print_last_validation_stats();
		//TODO: enter online classification mode prompt
	}
	else
	{
		std::println("failure");
		//return -1;
		std::println("New model will be created");
	}



	auto load_dataset = [&]<class DatasetType>(tag_holder<DatasetType>, const char* type, auto&& ...args) {
		std::print("Loading {} dataset... ", type);
		auto result = DatasetType(std::forward<decltype(args)>(args)...);
		std::println("{} items loaded", size(result));
		return result;
	};

	//auto datagen_func = gen_data_pair_circle_square;
	//auto val_dataset = load_dataset(tag_holder<stub_dataset>{}, "validation", datagen_func, 25);
	//auto train_dataset = load_dataset(tag_holder<stub_dataset>{}, "training", datagen_func, 250);
	auto val_dataset = load_dataset(tag_holder<dataset>{}, "validation", dataset_root / "val");
	auto train_dataset = load_dataset(tag_holder<dataset>{}, "trainning", dataset_root / "train");



	auto set_default_learning_rate = [&] {
		learning_rate_base = 0.1f;
		learning_rate_decay_rate = 1.1f;
		learning_rate_decay = 0;
	};
	auto create_model = [&] {
		m.set_input_size(input_size(train_dataset));

		m.add_layer(convolution_layer(3, 3, 10));
		m.add_layer(pooling_layer(2, 2));
		m.add_layer(tied_bias_layer{});
		m.add_layer(leaky_relu_layer{0.1f});

		m.add_layer(convolution_layer(3, 3, 10));
		m.add_layer(pooling_layer(2, 2));
		m.add_layer(tied_bias_layer{});
		m.add_layer(leaky_relu_layer{0.1f});

		m.add_layer(convolution_layer(3, 3, 10));
		m.add_layer(pooling_layer(2, 2));
		m.add_layer(tied_bias_layer{});
		m.add_layer(leaky_relu_layer{0.1f});

		m.add_layer(convolution_layer(3, 3, 10));
		m.add_layer(pooling_layer(2, 2));
		m.add_layer(tied_bias_layer{});
		m.add_layer(leaky_relu_layer{0.1f});

		m.add_layer(convolution_layer(3, 3, 10));
		m.add_layer(pooling_layer(2, 2));
		m.add_layer(tied_bias_layer{});
		m.add_layer(leaky_relu_layer{0.1f});

		m.add_layer(flattening_layer{});

		m.add_layer(dense_layer{ 16 });
		m.add_layer(untied_bias_layer{});
		m.add_layer(leaky_relu_layer{});

		m.add_layer(dense_layer{ 8 });
		m.add_layer(untied_bias_layer{});
		m.add_layer(leaky_relu_layer{});

		m.add_layer(dense_layer{ 2 });
		m.add_layer(untied_bias_layer{});
		m.add_layer(softmax_layer{});

		m.finish(cross_entropy_loss_function{});

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



	std::println("Model params: {}", m.get_total_parameter_count());
	std::println("Batch size: {}", batch_size);
	std::println("Learning rate: {}", learning_rate());



	bool report_progress = true;



	bool should_stop = false;
	bool should_enter_menu = false;

	auto menu = [&] {
		cursor_pos_holder cursor;
		size_t lines = 0;

		if (last_val_up_to_date)
		{
			print_last_validation_stats(); lines += 2;
		}
		std::println("Menu:"); ++lines;
		std::println(" learning rate control: + -");  ++lines;
		std::println(" batch size control: * / w s");  ++lines;
		std::println(" live fitting display control: .");  ++lines;
		std::println(" exit menu: (space)");  ++lines;
		std::println(" exit menu for 1 iteration: `");  ++lines;
		std::println(" exit program after exiting menu: ESC");  ++lines;
		std::println(" save formatted stats history to stats.txt: a");  ++lines;

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

			case 'a': save_stats(); break;

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



	fp best_val_loss = INFINITY;



	size_t epochs_passed = stats_history.size();
	const size_t epochs_limit = -1;
	const size_t epoch_evaluation_period = 1;



	auto print_stats = [&](stats_t stats) {
		std::print(" {:>10.7f}| {:>7.4f}|", stats.loss, stats.accuracy);
	};



	cursor_pos_holder cursor{ cursor_pos_holder::noacquire };



	const auto xclock = std::chrono::steady_clock::now;
	decltype(xclock() - xclock()) dt_sum{};



	auto step_evaluate = [&]() __declspec(noinline) {
		const bool eval_requested = epoch_evaluation_period != 0;
		const bool eval_period_passed = epochs_passed % epoch_evaluation_period == 0;

		if (!(eval_requested && eval_period_passed))
			return;

		cursor.acquire();
		last_stats.val = m.evaluate(val_dataset, pool, report_progress);
		cursor.release();
		last_val_up_to_date = true;
		clear_line();
	};

	auto step_save_gather = [&]() __declspec(noinline) {
		stats_history.push_back(last_stats);

		if (last_stats.val.loss < best_val_loss)
		{
			best_val_loss = last_stats.val.loss;
			save_model();
		}
	};

	size_t local_epochs_passed = 0;
	auto step_fit = [&]() __declspec(noinline) {

		cursor.acquire();

		auto t1 = xclock();
		last_stats.train = m.fit(train_dataset, pool, learning_rate(), batch_size, report_progress);
		auto t2 = xclock();
		
		cursor.release();
		clear_line();

		dt_sum += t2 - t1; 
		++local_epochs_passed;
	};

	auto step_print_helper = [&](size_t epoch, learn_statistics stats) {
		std::print("{:>6}|", epoch);
		print_stats(stats.train);
		print_stats(stats.val);
		std::println("");
		};
	auto step_print = [&]() __declspec(noinline) {
		step_print_helper(epochs_passed, last_stats);
	};

	auto step_advance = [&]() __declspec(noinline) {
		++epochs_passed;

		if (epochs_passed == epochs_limit)
			should_stop = true;

		last_val_up_to_date = false;
	};

	auto step_interactive = [&]() __declspec(noinline) {
		while (_kbhit()) if (_getch() == ' ') should_enter_menu = true;
		if (should_enter_menu) menu();
	};

	std::function<void()> steps[] = {
		step_interactive,
		step_evaluate,
		step_interactive,
		step_fit,
		step_print,
		step_save_gather,
		step_advance,
	};


	for (const auto& entry : stats_history)
		step_print_helper(idx_in(stats_history, entry), entry);


	while (true)
	{
		if (current_step == std::size(steps))
			current_step = 0;

		steps[current_step++]();

		if (should_stop)
			break; 
	}

	tui_bar();
	std::print("{}s/epoch\n", dt_sum.count() * 1e-9 / epochs_passed);
}
