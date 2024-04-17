
module;

#define NOMINMAX
#include <Windows.h>

#include <vector>
#include <numeric>
#include <ranges>

#include "defs.hpp"



export module diploma.utility;

export
{
#define X(num, name) constexpr size_t name ## _idx = num;
	image_classes_xlist;

	template<class T>
	using dynarray = std::vector<T>;
	using fp = float;

	std::string wide_to_narrow(const std::wstring& w)
	{
		const size_t mbsz =
			WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), nullptr, 0, nullptr, nullptr);
		std::string s(mbsz, ' ');
		WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), s.data(), (int)mbsz, nullptr, nullptr);
		return s;
	}

	int __tlregdtor()
	{
		//MSVC bug workaround
		return 0;
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

//#define X(num, name)

	std::string_view classification_to_string(const dynarray<fp>& stats)
	{
		xassert(stats.size() == 3, "Incorrect classification result vector of size {} (expected 3)", stats.size());
		switch (get_max_idx(stats))
		{

#define X(num, name) case num: return #name;
		image_classes_xlist;

		default:
			std::unreachable();
		}
	}
}
