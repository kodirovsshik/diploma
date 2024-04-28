
module;

#define NOMINMAX
#include <Windows.h>

#include <vector>
#include <numeric>
#include <ranges>
#include <random>

#include "defs.hpp"



export module diploma.utility;


EXPORT_BEGIN

using fp = float;

using cpath = const std::filesystem::path&;
using idx_t = std::ptrdiff_t;

constexpr size_t rng_seed = 1;


template<class T, class R>
void randomize_range(R&& range, T from, T to)
{
	static thread_local std::mt19937_64 thread_rng(rng_seed);

	std::uniform_real_distribution<T> distr(from, to);
	for (auto&& x : range)
		x = distr(thread_rng);
}

template<class T, class R>
size_t idx_in(const R& r, const T& x)
{
	return (size_t)(&x - &*r.begin());
}


std::string wide_to_narrow(const std::wstring& w)
{
	const size_t mbsz =
		WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), nullptr, 0, nullptr, nullptr);
	std::string s(mbsz, ' ');
	WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), s.data(), (int)mbsz, nullptr, nullptr);
	return s;
}

template<class T>
T safe_mul(T a, T b)
{
	if constexpr (DO_DEBUG_CHECKS)
	{
		if (a == 0 || b == 0)
			return 0;

		const auto max = std::numeric_limits<T>::max();

		if (max / b < a)
			throw;
		else if (max / b == a && max % b != 0)
			throw;
	}
	return a * b;
}
template<class T>
T safe_add(T a, T b)
{
	if constexpr (DO_DEBUG_CHECKS)
	{
		const auto max = std::numeric_limits<T>::max();

		if (a > max - b)
			throw;
	}
	return a + b;
}
template<class T>
T safe_fma(T a, T b, T c)
{
	return safe_add(safe_mul(a, b), c);
}

template<class R>
void normalize(R& range)
{
	const auto sum = std::accumulate(std::begin(range), std::end(range), std::ranges::range_value_t<R>{});
	for (auto& x : range)
		x /= sum;
}

template<class R>
size_t get_max_idx(const R& range)
{
	const auto enumerated_range = std::ranges::views::enumerate(range);
	const auto enumerated_range_to_value = [](const auto& p) { return std::get<1>(p); };
	const auto max_enumerated_pair = std::ranges::max(enumerated_range, {}, enumerated_range_to_value);
	return (size_t)std::get<0>(max_enumerated_pair);
}


EXPORT_END

