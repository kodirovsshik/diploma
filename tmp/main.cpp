
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



int main1()
{
	const std::filesystem::path base_dir = std::format("D:\\nn\\{}\\", rng_init_value);
	std::filesystem::create_directories(base_dir);

	uint64_t dt;

	thread_pool pool(8);
	nn_t nn_good = create_preset_topology_nn(), nn_pending, nn_new;
	nn_good.randomize(rng);

	const fp rate = 0.001f, decay = 1.05f;
	size_t decay_n = 0;
	size_t learn_n = 0;

	auto get_rate = [&] { return rate * powf(decay, -(fp)decay_n); };

	auto dataset = timeit([] { return read_main_dataset(); }, dt);
	std::print("dataset load dt = {} ms\n", dt / 1000000);
	std::ranges::shuffle(dataset, rng);

	bool reset_pending = true;
	fp good_cost;

	while (true)
	{
		if (reset_pending)
		{
			reset_pending = false;
			nn_pending = nn_good;
			good_cost = nn_apply_gradient_descend_iteration(nn_pending, dataset, learn_n++, pool, get_rate());
		}

		nn_new = nn_pending;
		const fp pending_cost = nn_apply_gradient_descend_iteration(nn_new, dataset, learn_n++, pool, get_rate());

		if (pending_cost < good_cost)
		{
			std::print("cost -> {} (d = {})\n", pending_cost, good_cost - pending_cost);

			std::swap(nn_good, nn_pending);
			std::swap(nn_new, nn_pending);
			good_cost = pending_cost;

			if (nn_good.write(base_dir / "a.nn"))
				continue;

			std::print("Failed to write to a.nn, trying b.nn ...");
			if (nn_good.write(base_dir / "b.nn"))
			{
				std::print("ok\n");
				continue;
			}

			std::print("failed to save, stopping now");
			return 1;
		}
		else
		{
			++decay_n;
			std::print("decreasing learning rate to {}\n", get_rate());
			reset_pending = true;
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


void init()
{
	setlocale(0, "");

	rng.seed(std::hash<size_t>{}(rng_init_value));
	rng.discard(4096);
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
