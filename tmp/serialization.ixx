
module;

#include <iostream>
#include <ranges>
#include <type_traits>

export module diploma.serialization;

import diploma.crc;



using sink = std::ostream&;
using CRC = crc64_calculator&;



template<class T>
concept trivial_type = std::is_trivially_copyable_v<T>;

template<class...>
constexpr bool always_false = false;

template<class R>
concept trivial_contiguous_range =
std::ranges::range<R> &&
std::contiguous_iterator<std::ranges::iterator_t<R>> &&
trivial_type<typename std::iterator_traits<std::ranges::iterator_t<R>>::value_type>;



bool serialize_memory_region(sink os, const void* ptr, size_t size, CRC crc)
{
	os.write((const char*)ptr, size);
	if (os)
	{
		crc.update(ptr, size);
		return true;
	}
	return false;
}
template<class T>
bool serialize_object_memory(sink os, const T& x, CRC crc)
{
	return serialize_memory_region(os, &x, sizeof(x), crc);
}



template<class T>
struct serialize_helper
{
	bool operator()(sink, const T&, CRC) const
	{
		static_assert(always_false<T>, "Invalid type for serialization");
		return false;
	}
};

template<trivial_type T>
struct serialize_helper<T>
{
	bool operator()(sink os, const T& x, CRC crc) const
	{
		return serialize_object_memory(os, x, crc);
	}
};

template<std::ranges::range R> requires(!trivial_type<R>)
struct serialize_helper<R>
{
	bool operator()(sink os, const R& r, CRC crc) const
	{
		const auto size = (size_t)std::ranges::distance(r);
		if (!serialize_helper<size_t>{}(os, size, crc))
			return false;

		if constexpr (trivial_contiguous_range<R>)
		{
			const auto& first_element = *r.begin();
			const size_t elem_count = std::ranges::distance(r);
			return serialize_memory_region(os, std::addressof(first_element), sizeof(first_element) * elem_count, crc);
		}
		else
		{
			for (auto& x : r)
			{
				const bool ok = serialize_helper<std::remove_cvref_t<decltype(x)>>{}(os, x, crc);
				if (!ok) return false;
			}
			return true;
		}
	}
};

template<class... Ts> requires(!trivial_type<std::tuple<Ts...>>)
struct serialize_helper<std::tuple<Ts...>>
{
	template<size_t level = 0>
	bool traverse_tuple(sink os, const std::tuple<Ts...>& t, CRC crc) const
	{
		if constexpr (level == sizeof...(Ts))
			return true;
		else
		{
			using T = std::tuple_element_t<level, std::tuple<Ts...>>;

			const T& val = std::get<level>(t);

			const bool ok = serialize_helper<T>{}(os, val, crc);
			if (!ok)
				return ok;
			return traverse_tuple<level + 1>(os, t, crc);
		}
	}
	bool operator()(sink os, const std::tuple<Ts...>& t, CRC crc) const
	{
		return this->traverse_tuple(os, t, crc);
	}
};



export class serializer_t
{
	crc64_calculator crc_;
	std::ostream& out;

public:
	serializer_t(std::ostream& os) : out(os) {}

	template<class T>
	bool operator()(const T& x)
	{
		return serialize_helper<T>{}(out, x, crc_);
	}

	bool write_crc() { return (*this)(this->crc()); }
	uint64_t crc() const { return this->crc_.value(); }

	explicit operator bool() const noexcept { return (bool)this->out; }
};
