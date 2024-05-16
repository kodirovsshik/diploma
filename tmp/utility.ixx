
export module diploma.utility;

#define NOMINMAX //why do you not work
import <Windows.h>;
#undef min
#undef max

import <vector>;
import <numeric>;
import <ranges>;
import <random>;

import "defs.hpp";



EXPORT_BEGIN

using fp = float;

using csw = const std::string_view;
using cs = const std::string&;
using path = std::filesystem::path;
using cpath = const path&;
using idx_t = std::ptrdiff_t;

constexpr size_t rng_seed_index = 1;
const size_t rng_seed = std::hash<size_t>{}(rng_seed_index);
thread_local std::mt19937_64 thread_rng(rng_seed);


template<class R>
void randomize_range(R&& range, std::ranges::range_value_t<R> from, std::ranges::range_value_t<R> to)
{
	std::uniform_real_distribution<std::ranges::range_value_t<R>> distr(from, to);
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

template<class T> const T nl_min = std::numeric_limits<T>::min();
template<class T> const T nl_max = std::numeric_limits<T>::max();
template<class T> constexpr bool nl_signed = std::numeric_limits<T>::is_signed;
template<class T> constexpr bool nl_bounded = std::numeric_limits<T>::is_bounded;

template<class T>
T safe_add(T a, T b)
{
	if constexpr (DO_DEBUG_CHECKS)
	{
		const auto max = nl_max<T>;
		const auto min = nl_min<T>;

		//TODO: fix
		bool ok = true;

		if (a > max - b)
			ok = false;

		xassert(ok, "overflow: adding {} and {}", a, b);
	}
	return a + b;
}
template<class T>
T safe_mul(T a, T b)
{
	if constexpr (DO_DEBUG_CHECKS)
	{
		if (a == 0 || b == 0)
			return 0;

		const auto max = nl_max<T>;
		const auto min = nl_min<T>;

		//TODO: fix
		bool ok = true;
		
		if (max / b < a)
			ok = false;
		else if (max / b == a && max % b != 0)
			ok = false;

		xassert(ok, "overflow: multiplying {} by {}", a, b);
	}
	return a * b;
}
template<class T>
T safe_fma(T a, T b, T c)
{
	return safe_add(safe_mul(a, b), c);
}
template<class T>
std::pair<T, T> safe_divmod(T a, T b)
{
	if constexpr (DO_DEBUG_CHECKS)
	{
		bool ok = true;

		//assumes two's complement
		if constexpr (nl_signed<T> && nl_bounded<T>)
			if (a == nl_max<T> && b == -1)
				ok = false;

		xassert(ok, "divmod error: {} / {} not respesentable (even with remainder) as {}", a, b, typeid(T).name());
	}
	return { a / b, a % b };
}
template<class T>
T safe_div(T a, T b)
{
	if constexpr (DO_DEBUG_CHECKS)
	{
		bool ok = true;

		auto [q, r] = safe_divmod(a, b);
		if (r) ok = false;

		xassert(ok, "division error: {} / {} not respesentable as {}", a, b, typeid(T).name());
		return q;
	}
	else
		return a / b;
}

template<typename Out, typename In>
Out int_cast(In x)
{
	bool ok = true;

#if DO_DEBUG_CHECKS
	if (!nl_signed<Out> && x < 0) ok = false;

	auto omin_v = nl_min<Out>;
	auto omax_v = nl_max<Out>;

	if (!nl_signed<In> && nl_signed<Out>)
		omin_v = std::max<Out>(omin_v, 0);

	if (x < omin_v) ok = false;
	if (x > omax_v) ok = false;

	xassert(ok, "int_cast: failed to cast {} from {} to {}", x, typeid(In).name(), typeid(Out).name());
#endif

	return Out{ x };
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

//MSVC bug workaround
int __tlregdtor()
{
	return 0;
}


EXPORT_END

