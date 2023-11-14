
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

int main1()
{
	xassert(false, "test");

	auto t1 = clock_f();
	auto dataset = read_main_dataset();
	auto t2 = clock_f();

	doNotOptimizeOut(dataset);

	std::print("dt = {}\n", std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1));

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
