
import std;
import libksn.math.linear_algebra;
import libksn.crc;

import <stdint.h>;
import <math.h>;
import <windows.h>;



template<class F, class Tuple, class... Args, std::size_t... I>
constexpr decltype(auto) apply_extra_impl(F&& f, Tuple&& t, std::index_sequence<I...>, Args&& ...args)
{
	return std::invoke(std::forward<F>(f), std::forward<Args>(args)..., std::get<I>(std::forward<Tuple>(t))...);
}
template<class F, class Tuple, class... Args>
constexpr decltype(auto) apply_extra(F&& f, Tuple&& t, Args&& ...args)
{
	return apply_extra_impl(
		std::forward<F>(f),
		std::forward<Tuple>(t),
		std::make_index_sequence<std::tuple_size_v<std::remove_cvref_t<Tuple>>>(),
		std::forward<Args>(args)...
	);
}



class thread_pool
{
	using func_t = void(void);
	using callback_t = std::move_only_function<func_t>;

	std::vector<std::thread> threads;
	std::queue<callback_t> tasks;
	std::mutex tasks_mutex;
	std::atomic_uint tasks_semaphore, free_threads_count;
	std::atomic_bool exit_flag;

	static constexpr unsigned int tasks_exit = -1;

	void worker()
	{
		while (true)
		{
			tasks_semaphore.wait(0);
			[] {}();

			callback_t f;
			tasks_mutex.lock();
			if (!tasks.empty())
			{
				--free_threads_count;
				f = std::move(this->tasks.front());
				this->tasks.pop();
				--tasks_semaphore;
			}
			tasks_mutex.unlock();

			if (f)
			{
				f();
				++free_threads_count;
				free_threads_count.notify_all();
			}
			else if (exit_flag)
			{
				return;
			}
		}
	}

	template<class F>
	void insert_task(F&& task)
	{
		this->tasks_mutex.lock();
		this->tasks.push(std::forward<F>(task));
		this->tasks_semaphore++;
		this->tasks_mutex.unlock();

		this->tasks_semaphore.notify_one();
	}


public:
	thread_pool(unsigned int n_threads = std::thread::hardware_concurrency())
	{
		for (unsigned int i = 0; i < n_threads; ++i)
			this->threads.emplace_back(&thread_pool::worker, this);
		free_threads_count = n_threads;
	}

	void barrier()
	{
		while (true)
		{
			const auto last_free_threads_count = this->free_threads_count.load();
			if (last_free_threads_count != this->threads.size())
				this->free_threads_count.wait(last_free_threads_count);

			std::scoped_lock lk(this->tasks_mutex);
			if (this->tasks_semaphore == 0 && this->free_threads_count == this->threads.size())
				return;
		}
	}

#define copy_as_tuple(args_) (std::tuple(std::forward<decltype(args_)>(args_)...)) //wordy

	template<class F, class... Ts>
	void schedule_task(F&& func, Ts&& ...args_)
	{
		auto caller = [&, args = copy_as_tuple(args_)]
		() mutable -> void
			{
				std::apply(
					std::forward<F>(func),
					std::forward<decltype(args)>(args)
				);
			};
		this->insert_task(std::move(caller));
	}
	template<class F, class... Ts>
	void schedule_ranged_task(uint64_t begin, uint64_t end, F&& func, Ts&& ...args_)
	{
		auto caller = [=, func = std::forward<F>(func), args = copy_as_tuple(args_)]
		() mutable -> void
			{
				for (size_t i = begin; i != end; ++i)
					apply_extra(
						std::forward<F>(func),
						std::forward<decltype(args)>(args),
						i
					);
			};
		this->insert_task(std::move(caller));
	}
	template<class F, class... Ts>
	void schedule_sized_work(const uint64_t begin, const uint64_t size, F&& func, Ts&& ...args_)
	{
		const size_t n = this->threads.size();
		const uint64_t split_size = size / n;
		const uint64_t leftover_size = size % n;

		uint64_t current_offset = begin;
		for (size_t i = 0; i < this->threads.size(); ++i)
		{
			const uint64_t current_size = split_size + bool(i < leftover_size);
			const uint64_t current_end = current_offset + current_size;
			this->schedule_ranged_task(current_offset, current_end, std::forward<F>(func), std::forward<Ts>(args_)...);
			current_offset = current_end;
		}
	}

	~thread_pool()
	{
		this->exit_flag = true;
		this->tasks_semaphore = thread_pool::tasks_exit;
		this->tasks_semaphore.notify_all();
		for (auto& th : this->threads)
			th.join();
	}
};




struct nn_layer_descriptor
{
	size_t output_image_count;
	size_t kernel_width;
	size_t kernel_height;
	size_t pooling_factor;
};

struct nn_image_stage_descriptor
{
	size_t count;
	size_t width;
	size_t height;
};

template<class T>
constexpr T safe_div(T dividend, T divisor)
{
	if (dividend % divisor) std::unreachable();
	return dividend / divisor;
}

namespace detail
{
	template<
		size_t layer,
		class fp_t,
		size_t _convolution_topology_size,
		std::array<nn_layer_descriptor, _convolution_topology_size> convolution_topology,
		nn_image_stage_descriptor input_image_descriptor,
		class... Kernels
	>
	consteval auto calculate_kernels_tuple_type()
	{
		if constexpr (layer == _convolution_topology_size)
			return std::type_identity<std::tuple<Kernels...>>{};
		else
		{
			constexpr size_t current_image_count = (layer == 0) ? input_image_descriptor.count : convolution_topology[layer - 1].output_image_count;
			constexpr size_t next_image_count = convolution_topology[layer].output_image_count;
			constexpr size_t kernels_count = current_image_count * next_image_count;

			using kernel_t = ksn::hmatrix<fp_t, convolution_topology[layer].kernel_height, convolution_topology[layer].kernel_width>;
			using new_kernel_array_t = std::array<kernel_t, kernels_count>;

			return calculate_kernels_tuple_type<layer + 1, fp_t, _convolution_topology_size, convolution_topology, input_image_descriptor, Kernels..., new_kernel_array_t>();
		}
	}

	template<
		size_t layer,
		class fp_t,
		size_t _convolution_topology_size,
		std::array<nn_layer_descriptor, _convolution_topology_size> convolution_topology,
		nn_image_stage_descriptor input_image_descriptor,
		class... Kernels
	>
	consteval auto calculate_offsets_tuple_type()
	{
		if constexpr (layer == _convolution_topology_size)
			return std::type_identity<std::tuple<Kernels...>>{};
		else
		{
			constexpr size_t offsets_count = convolution_topology[layer].output_image_count;
			using new_offset_array_t = ksn::hvector<fp_t, offsets_count>;

			return calculate_offsets_tuple_type<layer + 1, fp_t, _convolution_topology_size, convolution_topology, input_image_descriptor, Kernels..., new_offset_array_t>();
		}
	}

	template<
		class fp_t,
		size_t _convolution_topology_size,
		std::array<nn_layer_descriptor, _convolution_topology_size> convolution_topology,
		nn_image_stage_descriptor input_image_descriptor
	>
	using kernels_tuple_t = typename decltype(calculate_kernels_tuple_type<0, fp_t, _convolution_topology_size, convolution_topology, input_image_descriptor>())::type;
	
	template<
		class fp_t,
		size_t _convolution_topology_size,
		std::array<nn_layer_descriptor, _convolution_topology_size> convolution_topology,
		nn_image_stage_descriptor input_image_descriptor
	>
	using offsets_tuple_t = typename decltype(calculate_offsets_tuple_type<0, fp_t, _convolution_topology_size, convolution_topology, input_image_descriptor>())::type;



};




class crc64_calculator
{
	uint64_t crc = ksn::crc64_ecma_initial();

public:
	constexpr void update(const void* data, size_t size) noexcept
	{
		this->crc = ksn::crc64_ecma_update(data, size, crc);
	}
	constexpr uint64_t value() const noexcept
	{
		return this->crc;
	}
};

namespace detail
{
	template<class T>
	concept trivial_type = std::is_trivially_copyable_v<T>;


	using sink = std::ostream&;

	template<class...>
	constexpr bool always_false = false;

	template<class R>
	concept trivial_contiguous_range =
		std::ranges::range<R> &&
		std::contiguous_iterator<std::ranges::iterator_t<R>> &&
		trivial_type<typename std::iterator_traits<std::ranges::iterator_t<R>>::value_type>;


	bool serialize_memory_region(sink os, const void* ptr, size_t size, crc64_calculator& crc)
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
	bool serialize_object_memory(sink os, const T& x, crc64_calculator& crc)
	{
		return serialize_memory_region(os, &x, sizeof(x), crc);
	}

	template<class T>
	struct serialize_helper
	{
		bool operator()(sink, const T&, crc64_calculator&) const
		{
			static_assert(always_false<T>, "Invalid type for serialization");
			return false;
		}
	};
	

	template<trivial_type T>
	struct serialize_helper<T>
	{
		bool operator()(sink os, const T& x, crc64_calculator& crc) const
		{
			return serialize_object_memory(os, x, crc);
		}
	};
	template<std::ranges::range R> requires(!trivial_type<R> && !trivial_contiguous_range<R>)
	struct serialize_helper<R>
	{
		bool operator()(sink os, const R& r, crc64_calculator& crc) const
		{
			for (auto& x : r)
			{
				const bool ok = serialize_helper<std::remove_cvref_t<decltype(x)>>{}(os, x, crc);
				if (!ok) return false;
			}
			return true;
		}
	};
	template<trivial_contiguous_range R> requires(!trivial_type<R>)
	struct serialize_helper<R>
	{
		bool operator()(sink os, const R& r, crc64_calculator& crc) const
		{
			const auto& first_element = *r.begin();
			const size_t elem_count = std::ranges::distance(r);
			return serialize_memory_region(os, std::addressof(first_element), sizeof(first_element) * elem_count, crc);
		}
	};
	template<class... Ts> requires(!trivial_type<std::tuple<Ts...>>)
	struct serialize_helper<std::tuple<Ts...>>
	{
		template<size_t level = 0>
		bool traverse_tuple(sink os, const std::tuple<Ts...>& t, crc64_calculator& crc) const
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
		bool operator()(sink os, const std::tuple<Ts...>& t, crc64_calculator& crc) const
		{
			return traverse_tuple(os, t, crc);
		}
	};
}

class serializer_t
{
	crc64_calculator crc_;
	std::ostream& out;

public:
	serializer_t(std::ostream& os) : out(os) {}

	template<class T>
	bool operator()(const T& x)
	{
		return detail::serialize_helper<T>{}(out, x, crc_);
	}

	bool write_crc() { return (*this)(this->crc()); }
	uint64_t crc() const { return this->crc_.value(); }

	explicit operator bool() const noexcept { return (bool)this->out; }
};





template<
	size_t _convolution_topology_size,
	std::array<nn_layer_descriptor, _convolution_topology_size> convolution_topology,
	nn_image_stage_descriptor input_image_descriptor
>
class nn_t
{
private:
	static constexpr uint64_t signature = 0x52EEF7E17876A668;


public:
	using fp_t = float;


	template<size_t stage>
	consteval static auto calculate_image_stage()
	{
		if (stage > 0 && stage - 1 >= _convolution_topology_size)
			std::unreachable();

		nn_image_stage_descriptor image = input_image_descriptor;
		for (size_t i = 0; i < stage; ++i)
		{
			image.width = safe_div(image.width - convolution_topology[i].kernel_width + 1, convolution_topology[i].pooling_factor);
			image.height = safe_div(image.height - convolution_topology[i].kernel_height + 1, convolution_topology[i].pooling_factor);
		}
		if (stage != 0) image.count = convolution_topology[stage - 1].output_image_count;
		return image;
	}


	detail::kernels_tuple_t<fp_t, _convolution_topology_size, convolution_topology, input_image_descriptor> kernels;
	detail::offsets_tuple_t<fp_t, _convolution_topology_size, convolution_topology, input_image_descriptor> offsets;


	bool write(std::ostream& out)
	{
		serializer_t serializer(out);
		serializer(this->signature);

		serializer(input_image_descriptor);
		serializer(convolution_topology);
		serializer.write_crc();

		serializer(this->kernels);
		serializer(this->offsets);
		serializer.write_crc();

		return (bool)serializer;
	}
	void write(const std::filesystem::path& p)
	{
		std::ofstream fout(p, std::ios_base::out | std::ios_base::binary);
		const bool ok = this->write(fout);
		if (!ok)
			std::print("Faield to serialize to {}\n", p.string());
	}
	void read(std::istream& in)
	{

	}
};

constexpr auto create_preset_convolution_topology_nn()
{
	constexpr nn_image_stage_descriptor input{ 3, 256, 256 };
	constexpr nn_layer_descriptor l0{ 5, 5, 5, 2 };
	constexpr nn_layer_descriptor l1{ 8, 3, 3, 2 };
	constexpr nn_layer_descriptor l_last{ 1, 3, 3, 1 };
	constexpr std::array convolution_topology{ l0, l1, l_last };
	return nn_t<convolution_topology.size(), convolution_topology, input>();
}


int main()
{
	auto nn = create_preset_convolution_topology_nn();
	nn.write("a.txt");
}
