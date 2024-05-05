
export module diploma.serialization;

import diploma.crc;

import <iostream>;
import <ranges>;
import <type_traits>;

import "defs.hpp";



using sink = std::ostream&;
using source = std::istream&;
using CRC = crc64_calculator&;



template<class T>
concept trivially_serializeable = std::is_trivially_copyable_v<T>;

template<class...>
constexpr bool always_false = false;

template<class R>
concept trivially_serializeable_range =
	std::ranges::range<R> &&
	std::contiguous_iterator<std::ranges::iterator_t<R>> &&
	trivially_serializeable<typename std::ranges::range_value_t<R>>
;



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

bool deserialize_memory_region(source is, void* ptr, size_t size, CRC crc)
{
	is.read((char*)ptr, size);
	if (is)
	{
		crc.update(ptr, size);
		return true;
	}
	return false;
}
template<class T>
bool deserialize_object_memory(source is, T& x, CRC crc)
{
	return deserialize_memory_region(is, (void*)&x, sizeof(x), crc);
}





#define strategy_list X(trivial_copy) X(range)

template<class T>
consteval auto pick_serialize_strategy();


#define X(strategy) friend struct serialize_strategy_ ## strategy;

export class serializer_t
{
	strategy_list;

	crc64_calculator crc_;
	std::ostream& out;

	template<class T>
	static constexpr bool has_serialize_method =
		requires(const T & t, serializer_t serializer) { { t.serialize(serializer) }; };

	template<class T>
	bool write(const T& x)
	{
		return serialize_object_memory(out, x, crc_);
	}
	bool write(const void* data, size_t data_bytes)
	{
		return serialize_memory_region(out, data, data_bytes, crc_);
	}

public:
	serializer_t(std::ostream& os) : out(os) {}

	template<class T> requires(!has_serialize_method<T>)
	bool operator()(const T& x)
	{
		return pick_serialize_strategy<T>()(x, *this);
	}

	template<class T> requires(has_serialize_method<T>)
	bool operator()(const T& x)
	{
		x.serialize(*this);
		return (bool)*this;
	}

	bool write_crc() { return (*this)(this->crc()); }
	uint64_t crc() const { return this->crc_.value(); }

	explicit operator bool() const noexcept { return (bool)this->out; }
};

export class deserializer_t
{
	strategy_list;

	template<class T>
	static constexpr bool has_deserialize_method =
		requires(T & t, deserializer_t deserializer) { { t.deserialize(deserializer) }; };

	crc64_calculator crc_;
	std::istream& in;

	template<class T>
	bool read(T& x)
	{
		return deserialize_object_memory(in, x, crc_);
	}
	bool read(void* data, size_t data_bytes)
	{
		return deserialize_memory_region(in, data, data_bytes, crc_);
	}

public:
	deserializer_t(std::istream& is) : in(is) {}

	template<class T> requires(!std::is_const_v<T>&& !has_deserialize_method<T>)
	bool operator()(T& x)
	{
		return pick_serialize_strategy<T>()(x, *this);
	}

	template<class T> requires(!std::is_const_v<T> && has_deserialize_method<T>)
	bool operator()(T& x)
	{
		return x.deserialize(*this) && (bool)*this;
	}

	bool test_crc()
	{
		const uint64_t expected_crc = this->crc();

		uint64_t observed_crc;
		if (!(*this)(observed_crc))
			return false;

		return observed_crc == expected_crc;
	}
	uint64_t crc() const { return this->crc_.value(); }

	explicit operator bool() const noexcept { return (bool)this->in; }
};





struct serialize_strategy_trivial_copy
{
	template<trivially_serializeable T>
	bool operator()(const T& x, serializer_t& serializer) const
	{
		return serializer.write(x);
	}
	template<trivially_serializeable T>
	bool operator()(T& x, deserializer_t& deserializer) const
	{
		return deserializer.read(x);
	}
};

struct serialize_strategy_range
{
	template<std::ranges::range R>
	bool operator()(const R& r, serializer_t& serializer) const
	{
		if (!serializer((size_t)std::ranges::distance(r)))
			return false;

		if constexpr (trivially_serializeable_range<R>)
		{
			const auto& first_element = *r.begin();
			const size_t elem_count = std::ranges::distance(r);
			return serializer.write(std::addressof(first_element), sizeof(first_element) * elem_count);
		}
		else
		{
			for (auto& x : r)
			{
				if (!serializer(x))
					return false;
			}
			return true;
		}
	}
	template<std::ranges::range R>
	bool operator()(R& r, deserializer_t& deserializer) const
	{
		size_t size;
		if (!deserializer(size))
			return false;

		if constexpr (requires() { r.resize(size); })
			r.resize(size);
		else if (std::ranges::distance(r) == size) {}
		else
			xassert(false, "Range size mismatch on deserialization");

		if constexpr (trivially_serializeable_range<R>)
		{
			const auto& first_element = *r.begin();
			return deserializer.read((void*)std::addressof(first_element), sizeof(first_element) * size);
		}
		else
		{
			for (auto& x : r)
			{
				if (!deserializer(x))
					return false;
			}
			return true;
		}
	}
};


template<class T>
consteval auto pick_serialize_strategy()
{
	if constexpr (std::ranges::range<T>)
		return serialize_strategy_range{};
	else if constexpr (trivially_serializeable<T>)
		return serialize_strategy_trivial_copy{};
	else
		static_assert(always_false<T>, "Invalid type for (de)serialization");
}
