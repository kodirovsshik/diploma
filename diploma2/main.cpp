
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

model m;


cpath dataset_root = R"(C:\dataset_pneumonia\bmp)";
cs model_codename = "test_cs3t5d8d2_ce";
cpath model_path = dataset_root / (model_codename + ".bin");
cpath stats_path = dataset_root / (model_codename + ".stats");

constexpr bool force_loaded_model = false;
constexpr bool force_no_existing_model = true;
constexpr bool force_override_model = false;

const size_t custom_epochs_limit = 151;
size_t epochs_limit = 151;
const size_t epoch_evaluation_period = 1;



int main()
{
	//main1(); return 0;

	fp learning_rate_base;
	fp learning_rate_decay_rate;
	int learning_rate_decay;

	auto learning_rate = [&] { return learning_rate_base * powf(learning_rate_decay_rate, -(fp)learning_rate_decay); };

	auto set_default_learning_rate = [&] {
		learning_rate_base = 0.1f;
		learning_rate_decay_rate = 1.1f;
		learning_rate_decay = 7;
		};



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
	auto save_stats = [&] (bool echo = false)
	{
		std::ofstream fout(stats_path);
		if (!fout.is_open())
		{
			std::println("Failed to save stats to {}", (char*)stats_path.generic_u8string().data());
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
		if (echo)
			std::println("Stats saved to {}", (char*)stats_path.generic_u8string().data());
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

		save_stats();
	};



	size_t current_step = 0;



	std::println("Dataset root: {}", (char*)dataset_root.generic_u8string().data());
	std::print("Trying to load model {} ... ", model_codename);
	if (!force_override_model && try_load_model())
	{
		std::println("success");
		if constexpr (force_no_existing_model)
		{
			std::println("Exiting due to force_no_existing_model=1");
			return -1;
		}
		print_last_validation_stats();
		//TODO: enter online classification mode prompt
	}
	else
	{
		if constexpr (force_override_model)
			std::println("attempt skipped due to force_override_model");
		else
			std::println("failure");

		if constexpr (force_loaded_model)
		{
			std::println("Exiting due to force_loaded_model=1");
			return -1;
		}
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



	size_t epochs_passed = stats_history.size();



	bool report_progress = true;



	bool should_stop = epochs_limit == epochs_passed;
	bool should_enter_menu = should_stop;

	auto menu = [&] {
		cursor_pos_holder cursor;
		size_t lines = 0;
		size_t last_epochs_limit = epochs_limit;

		if (last_val_up_to_date)
		{
			print_last_validation_stats(); lines += 2;
		}
		std::println("Learning_rate: {} ", learning_rate()); ++lines;
		std::print("Epochs limit: "); ++lines;
		if (epochs_limit == -1)
			std::println("inf");
		else
			std::println("{}", epochs_limit);

		std::println("Menu:"); ++lines;
		std::println(" epoch limit size control: e d");  ++lines;
		std::println(" epoch limit presence control: r f");  ++lines;
		std::println(" learning rate control: + -");  ++lines;
		std::println(" batch size control: * / w s");  ++lines;
		std::println(" live fitting display control: .");  ++lines;
		std::println(" save formatted stats history to stats.txt: a");  ++lines;
		std::println(" exit program after exiting menu: ESC");  ++lines;
		std::println(" exit menu for 1 iteration: `");  ++lines;
		std::println(" exit menu: (space)");  ++lines;

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

		auto mutate_epochs_limit = [&](int delta) {
			if (epochs_limit == -1)
			{
				std::println("Epoch limit is not set");
				return;
			}
			mutate_var(epochs_limit, epochs_limit + delta);
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
			case 'e': mutate_epochs_limit(+1); break;
			case 'd': mutate_epochs_limit(-1); break;
			case 'r': mutate_var(epochs_limit, last_epochs_limit == -1 ? custom_epochs_limit : last_epochs_limit); break;
			case 'f': mutate_var(epochs_limit, -1); break;
			case '*': mutate_var(batch_size, max_batch_size); break;
			case '/': mutate_var(batch_size, small_batch_size); break;
			case '.': mutate_var(report_progress, !report_progress); break;

			case '+': mutate_learning_rate(+1); break;
			case '-': mutate_learning_rate(-1); break;

			case 'a': save_stats(true); break;

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

	auto step_update_stats_history = [&]() __declspec(noinline) {
		stats_history.push_back(last_stats);
	};

	auto step_save_model = [&]() __declspec(noinline) {
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

		last_val_up_to_date = false;

		if (epochs_passed == epochs_limit)
			should_stop = true;
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
		step_update_stats_history,
		step_save_model,
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
	std::print("{}s/epoch\n", dt_sum.count() * 1e-9 / (local_epochs_passed ? local_epochs_passed : 1));
}
