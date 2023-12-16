
import diploma.crc;
import diploma.bmp;
import diploma.thread_pool;
import diploma.serialization;
import diploma.nn;

#include <locale.h>
#include <stdint.h>

#include <vector>
#include <random>
#include <chrono>
#include <filesystem>
#include <print>
#include <fstream>
#include <stacktrace>

#include <Windows.h>

#include "defs.hpp"

#undef min
#undef max


const auto clock_f = std::chrono::steady_clock::now;
using duration_t = decltype(clock_f() - clock_f());

std::mt19937_64 rng;
const size_t rng_init_value = 3;



template<class T>
void doNotOptimizeOut(const volatile T& obj)
{
	static volatile char x = 0;
	x = *(volatile char*)std::launder(std::addressof(obj));
}

template<class F>
decltype(auto) timeit(F&& f, uint64_t& dt)
{
	if constexpr (std::is_void_v<std::remove_cv_t<std::invoke_result_t<F&&>>>)
	{
		const auto t1 = clock_f();
		f();
		const auto t2 = clock_f();

		dt = (t2 - t1).count();
		return;
	}
	else
	{
		const auto t1 = clock_f();
		decltype(auto) result = f();
		const auto t2 = clock_f();

		dt = (t2 - t1).count();
		doNotOptimizeOut(result);
		return result;
	}
}



volatile bool stop_switch = false;
volatile bool stop_on_save_switch = false;
BOOL console_ctrl_handler(DWORD val)
{
	switch (val)
	{
	case CTRL_C_EVENT:
		stop_on_save_switch = true;
		return true;

	case CTRL_BREAK_EVENT:
		stop_switch = true;
		return true;

	default:
		return false;
	}
}



std::string wide_to_narrow(const std::wstring& w)
{
	const size_t mbsz =
		WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), nullptr, 0, nullptr, nullptr);
	std::string s(mbsz, ' ');
	WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), s.data(), (int)mbsz, nullptr, nullptr);
	return s;
}


#define wprint(fmt, ...) { fputws(std::format(fmt __VA_OPT__(,) __VA_ARGS__).data(), stdout); }

int __tlregdtor()
{
	//MSVC bug workaround
	return 0;
}

int online_classification(const nn_t& nn)
{
	static wchar_t buffer[512];

	OPENFILENAMEW info{};
	info.lStructSize = sizeof(info);
	info.hwndOwner = GetConsoleWindow();
	info.lpstrFilter = L"Compatible BMP Images (.bmp)\0*.bmp\0\0";
	info.lpstrFile = buffer;
	info.nMaxFile = sizeof(buffer) / sizeof(buffer[0]);
	info.lpstrTitle = L"Select an image for classification";
	info.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;

	while (true)
	{
		buffer[0] = 0;
		const auto ok = (bool)GetOpenFileNameW(&info);
		if (!ok && CommDlgExtendedError() == 0)
			return 0;

		bmp_image img;
		try
		{
			img.reset();
			img.read(buffer, true);
			if (img.width != 256 || img.height != 256)
				throw std::wstring(L"Invalid image size");

			auto& data = img.planes[0];
			nn.eval(data);
			auto sum = data[0] + data[1];
			std::print("Data: {}, {} (status: {})\n",
				data[0] / sum, data[1] / sum,
				data[0] > data[1] ? "positive" : "negative"
			);
		}
		catch (const std::wstring& err)
		{
			wprint(L"{} error: \"{}\": {}\nPress any key to continue\n",
				img.n_planes == 0 ? L"File read" : L"Classification", buffer, err
			);
		}
		while (_getch() != ' ');
	}
}

int main1()
{
	cpath base_dir = std::format("D:\\nn\\{}\\", rng_init_value),
		nn_path = base_dir / "a.nn";

	std::filesystem::create_directories(base_dir);


	uint64_t dt;


	nn_t nn_good = create_preset_topology_nn(), nn_pending, nn_new;

	fp rate, decay;
	int64_t decay_n;
	auto get_rate = [&] { return rate * powf(decay, -(fp)decay_n); };


	auto serialize_state = [&]
		{
			std::ofstream fout(nn_path, std::ios::out | std::ios::binary);
			serializer_t out(fout);

			out(rate);
			out(decay);
			out(decay_n);
			out.write_crc();
			nn_good.write(out);

			return (bool)out;
		};
	auto deserialize_state = [&]() -> bool
		{
			std::ifstream fin(nn_path, std::ios::in | std::ios::binary);
			deserializer_t in(fin);

			in(rate);
			in(decay);
			in(decay_n);
			rfassert(in.test_crc());
			nn_good.read(in);

			return (bool)in;
		};
	auto deserialize_or_init = [&]
		{
			std::print("Session seed is {}\n", rng_init_value);
			std::print("Trying to load existing model... ");
			if (deserialize_state())
			{
				std::print("success\n");
				return;
			}

			std::print("no existing model found\nAll NN parameters set to random\nAll hyperparameters set to default values\n");
			nn_good.randomize(rng);
			rate = 0.001f;
			decay = 1.1f;
			decay_n = 0;
		};

	deserialize_or_init();


	std::print("Enter online classification mode? (y/n):\n");
	while (true)
	{
		const char c = _getch();
		if (c == 'y' || c == 'Y')
			return online_classification(nn_good);
		if (c == 'n' || c == 'N')
			break;
	}


	thread_pool pool;

	std::print("loading training dataset... ");
	auto dataset = timeit([] { return read_training_dataset(); }, dt);
	std::print("size = {}, dt = {} ms\n", dataset.size(), dt / 1000000);

	std::print("loading validation dataset... ");
	const auto validation_dataset = timeit([] { return read_test_dataset(); }, dt);
	std::print("size = {}, dt = {} ms\n", validation_dataset.size(), dt / 1000000);



#if 0 //stochastic
	const size_t max_stochastic_batch = 1000;
#else
	const size_t max_stochastic_batch = dataset.size();
#endif
	auto stochastic_shuffle = [&]
		{
			if (max_stochastic_batch == dataset.size())
				return;

			static std::mt19937_64 shuffle_rng(rng_init_value);
			for (size_t i = 0; i < max_stochastic_batch; ++i)
				std::swap(dataset[i], dataset[shuffle_rng() % (dataset.size() - i) + i]);
		};



	std::print("evaluating initial cost... ");
	auto good_cost = nn_eval_cost(nn_good, dataset, pool).first;
	std::print("cost = {}\n", good_cost);

	{
		const auto good_stats = nn_eval_cost(nn_good, validation_dataset, pool).second;
		std::print("AC = {:.3g}%, TP ~ {:.3g}%, TN ~ {:.3g}%, FP ~ {:.3g}%, FN ~ {:.3g}%\n",
			good_stats.accuracy() * 100,
			good_stats.tp_frac() * 100, good_stats.tn_frac() * 100,
			good_stats.fp_frac() * 100, good_stats.fn_frac() * 100
		);
	}


	auto gradient_descend = [&]
	(nn_t& nn)
		{
			stochastic_shuffle();
			return nn_apply_gradient_descend_iteration(nn, dataset, max_stochastic_batch, pool, get_rate());
		};


	std::print("Training is running on {} threads\n", pool.size());
	std::print("Using {} gradient descend\n", max_stochastic_batch == dataset.size() ? "regular" : "stochastic");
	std::print("Learning rate is {}\n", get_rate());
	std::print("Press Ctrl-C to finish\nPress Ctrl-Break to cancel\n");

	bool enter_settings = true;
	bool continue_on_increased_cost = true;

	auto settings_menu = [&]
		{
			enter_settings = false;
			std::print(" Settings mode:\n");
			std::print(" rate = {} (decay_n = {}) (q = up, a = down)\n", get_rate(), decay_n);
			std::print(" continue_on_increased_cost = {} (w = change)\n", continue_on_increased_cost);
			std::print(" ESC = continue\n");
			std::print(" ~ = continue for one iteration\n");

			while (true)
			{
				bool exit = false;
				switch (_getch())
				{
				case 'q':
					--decay_n;
					std::print(" Increasing learning rate by user input: {} (n = {})\n", get_rate(), decay_n);
					break;
				case 'a':
					++decay_n;
					std::print(" Decreasing learning rate by user input: {} (n = {})\n", get_rate(), decay_n);
					break;

				case 'w':
					continue_on_increased_cost = !continue_on_increased_cost;
					std::print(" continue_on_increased_cost changed to {}\n", continue_on_increased_cost);
					break;

				case '`':
					enter_settings = true;
				case 27:
					exit = true;
					break;
				}

				if (exit)
					break;
			};
		};


	bool reset_pending = true;

	size_t iteration = 0;
	while (true)
	{
		if (stop_switch)
		{
			std::print("Unconditionally stopping by user request\n");
			break;
		}


		const bool have_focus = GetForegroundWindow() == GetConsoleWindow();

		if (enter_settings || have_focus && GetAsyncKeyState(VK_NUMPAD0) < 0)
			settings_menu();

		std::print("Iteration {}: ", iteration++);

		if (reset_pending)
		{
			reset_pending = false;
			nn_pending = nn_good;
			gradient_descend(nn_pending);
		}

		nn_new = nn_pending;
		const fp pending_cost = timeit([&] { return gradient_descend(nn_new); }, dt).first;

		if (continue_on_increased_cost || pending_cost < good_cost)
		{
			const auto pending_stats = nn_eval_cost(nn_pending, validation_dataset, pool).second;
			std::print("{} ms, cost -> {} (d = {}), AC = {}%, FP ~ {}%, FN ~ {}%\n",
				dt / 1000000, pending_cost, good_cost - pending_cost, pending_stats.accuracy() * 100, pending_stats.fp_frac() * 100, pending_stats.fn_frac() * 100);

			std::swap(nn_good, nn_pending);
			std::swap(nn_new, nn_pending);
			good_cost = pending_cost;

			bool save_ok = serialize_state();

			if (!save_ok)
			{
				std::print("Save failure, stopping by error");
				return 1;
			}

			if (!stop_on_save_switch)
				continue;

			std::print("Save complete, stopping by user request\n");
			break;
		}
		else
		{
			++decay_n;
			std::print("cost -> {} (+{}), learning rate -> {}\n", pending_cost, pending_cost - good_cost, get_rate());
			reset_pending = true;
		}
	}

	return 0;
}





void init()
{
	auto loc = setlocale(0, "");

	rng.seed(std::hash<size_t>{}(rng_init_value));
	rng.discard(4096);

	SetConsoleCtrlHandler(console_ctrl_handler, true);
}

void print_stacktrace()
{
	std::print("STACK TRACE:\n{}\n", std::stacktrace::current());
}

int main()
{
	try
	{
		init();
		return main1();
	}
	catch (const std::filesystem::filesystem_error& e)
	{
		std::print("EXCEPTION std::filesystem::filesystem_error\nwhat() = {}\n", e.what());
		wprint(L"path1() = {}\npath2 = {}\n", e.path1().wstring(), e.path2().wstring());
		print_stacktrace();
	}
	catch (const std::exception& e)
	{
		std::print("EXCEPTION std::exception\nwhat() = {}\n", e.what());
		print_stacktrace();
	}
	catch (const std::wstring& str)
	{
		wprint(L"EXCEPTION std::wstring\nwhat() = {}\n", str);
		print_stacktrace();
	}
	catch (...)
	{
		std::print("EXCEPTION (unknown type)\n");
		print_stacktrace();
	}

	return -1;
}
