
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

#include <Windows.h>

#include "defs.hpp"

#undef min
#undef max


const auto clock_f = std::chrono::steady_clock::now;
using duration_t = decltype(clock_f() - clock_f());

std::mt19937_64 rng;



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



int main1()
{
	uint64_t dt;

	thread_pool pool;
	nn_t nn1 = create_preset_topology_nn(), nn2;
	nn1.randomize(rng);

	const fp rate = 0.001f, decay = 0.1f;
	size_t decay_n = 0;
	size_t learn_n = 0;

	auto get_rate = [&] { return rate / (1 + decay_n * decay); };

	auto dataset = timeit([] { return read_main_dataset(); }, dt);
	std::print("dataset load dt = {} ms\n", dt / 1000000);
	std::ranges::shuffle(dataset, rng);

	fp old_cost = nn_eval_cost(nn1, dataset, pool);
	while (true)
	{
		nn2 = nn1;
		nn_apply_gradient_descend_iteration(nn2, dataset, learn_n, pool, get_rate());

		const fp new_cost = nn_eval_cost(nn2, dataset, pool);
		std::print("cost: {} -> {} (dcost = {})", old_cost, new_cost, old_cost - new_cost);
		if (new_cost > old_cost)
		{
			decay_n++;
			std::print("... reverting, decreasing learning rate to {}\n", get_rate());
		}
		else
		{
			std::print("\n");

			learn_n++;
			nn1 = std::move(nn2);
			old_cost = new_cost;;

			if (nn1.write("D:\\a.nn"))
				continue;

			std::print("Failed to write to a.nn, trying again ...");
			if (nn1.write("D:\\a.nn"))
			{
				std::print("ok\n");
				continue;
			}

			std::print("Failed to write to a.nn, trying b.nn ...");
			if (nn1.write("D:\\b.nn"))
			{
				std::print("ok\n");
				continue;
			}

			std::print("failed to save, stopping now");
			return 1;
		}
	}

	return 0;
}





std::string wide_to_narrow(const std::wstring& w)
{
	const size_t mbsz = WideCharToMultiByte(CP_ACP, 0, w.data(), (int)w.size(), nullptr, 0, nullptr, nullptr);
	std::string s(mbsz, ' ');
	WideCharToMultiByte(CP_ACP, 0, w.data(), (int)w.size(), s.data(), (int)mbsz, nullptr, nullptr);
	return s;
}

void setup_rng()
{
	const volatile int seed = 1;
	rng.seed(std::hash<int>{}((int)seed));
	rng.discard(4096);
}

void init()
{
	setlocale(0, "");
	setup_rng();
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
		std::print("EXCEPTION std::filesystem::filesystem_error\nwhat() = {}\npath1() = ", e.what(), e.path1().string());
	}
	catch (const std::exception& e)
	{
		std::print("EXCEPTION std::exception\nwhat() = {}", e.what());
	}
	catch (const std::wstring& str)
	{
		std::print("EXCEPTION std::wstring\nwhat() = {}", wide_to_narrow(str));
	}
	catch (...)
	{
		std::print("EXCEPTION (unknown type)");
	}
	return -1;
}
