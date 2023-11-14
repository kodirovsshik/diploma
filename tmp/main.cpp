
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

template<class T>
void doNotOptimizeOut(const volatile T& obj)
{
	static volatile char x = 0;
	x = *(volatile char*)std::launder(std::addressof(obj));
}

std::mt19937_64 rng;


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
	thread_pool pool;
	auto nn = create_preset_topology_nn();

	uint64_t dt1, dt2;
	auto dataset = timeit([] { return read_main_dataset(); }, dt1);
	fp eval = timeit([&] { return nn_eval_over_dataset(nn, dataset, pool); }, dt2);

	std::print("dt1 = {} ms\n", dt1 / 1000000);
	std::print("dt2 = {} ms\n", dt2 / 1000000);
	std::print("eval = {}\n", eval);

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
