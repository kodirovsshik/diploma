
import std;
import libksn.crc;

import <stdint.h>;
import <math.h>;
import <windows.h>;

#undef min
#undef max


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





template<class T>
using dynarray = std::vector<T>;

using fp = float;
using vector = dynarray<fp>;
using matrix = dynarray<vector>;

#define xassert(cond, fmt, ...) { auto sc = std::source_location::current(); \
	if (!(cond)) \
	{\
		std::print("{}:{}: ASSERTION FAILED:\n", sc.file_name(), sc.line()); \
		std::print(fmt, __VA_ARGS__); \
		std::cin.get(); std::exit(-1); \
	} \
}

void multiply_mat_vec(const matrix& m, const vector& v, vector& x)
{
	if (m.size() == 0) 
	{
		x.clear();
		return;
	}

	const size_t n1 = m.size();
	const size_t n2 = m[0].size();
	xassert(n2 == v.size(), "incompatible multiplication: {}x{} by {}", n1, n2, v.size());

	x.resize(n1);
	for (size_t i = 0; i < n1; ++i)
		for (size_t j = 0; j < n2; ++j)
			x[i] += m[i][j] * v[j];
}
void add_vec_vec(const vector& x, vector& y)
{
	const size_t n1 = x.size();
	const size_t n2 = y.size();
	xassert(n1 == n2, "incompatible addition: {} and {}", n1, n2);

	for (size_t i = 0; i < n1; ++i)
		y[i] += x[i];
}



struct activation_function_data
{
	using func_ptr = fp(*)(fp);
	
	func_ptr f, df;
};

fp sigma(fp x)
{
	return 1 / (1 + ::expf(-x));
}
fp sigma_d(fp x)
{
	const fp s = sigma(x);
	return s * (1 - s);
}

activation_function_data sigmoid{ sigma, sigma_d };

#define lambda(input, val) [](input){ return (val); }

constexpr fp leaky_relu_param = (fp)0.01;
activation_function_data leaky_relu{
	lambda(fp x, std::max(x, x * leaky_relu_param)),
	lambda(fp x, (x < 0) ? leaky_relu_param : 1),
};


void activate_vector(vector& x, const activation_function_data& activator)
{
	for (auto& val : x)
		val = activator.f(val);
}


class nn_t
{
private:
	static constexpr uint64_t signature = 0x52EEF7E17876A668;


public:
	static constexpr size_t input_neurons = 65536;


	static void init_vector(vector& arr, size_t n)
	{
		arr.clear();
		arr.resize(n);
	}

	static void init_matrix(matrix& mat, size_t rows, size_t columns)
	{
		mat.resize(rows);
		for (auto& row : mat)
			init_vector(row, columns);
	}
	
	dynarray<matrix> weights;
	dynarray<vector> biases;
	dynarray<activation_function_data> activators;

	void create(std::span<const unsigned> layer_sizes)
	{
		const size_t n = layer_sizes.size();
		xassert(n >= 2, "Degererate NN topology with {} layers", n + 1);

		this->reset();

		weights.resize(n);
		biases.resize(n);
		activators.reserve(n);

		for (size_t i = 0; i < n - 1; ++i)
			activators.push_back(leaky_relu);
		activators.push_back(sigmoid);

		for (size_t i = 0; i < n; ++i)
			init_vector(biases[i], layer_sizes[i]);

		size_t prev_layer_size = input_neurons;
		for (size_t i = 0; i < n; ++i)
		{
			const size_t next_layer_size = layer_sizes[i];
			init_matrix(weights[i], next_layer_size, prev_layer_size);
			prev_layer_size = next_layer_size;
		}
	}


	bool write(std::ostream& out)
	{
		serializer_t serializer(out);
		serializer(this->signature);

		std::vector<size_t> topology(this->biases.size() + 1);
		topology[0] = this->input_neurons;
		for (size_t i = 0; i < this->biases.size(); ++i)
			topology[i + 1] = this->biases[i].size();
		
		serializer(topology);
		serializer.write_crc();

		topology = decltype(topology)();

		serializer(this->weights);
		serializer(this->biases);
		serializer.write_crc();

		return (bool)serializer;
	}
	void write(const std::filesystem::path& p)
	{
		std::ofstream fout(p, std::ios_base::out | std::ios_base::binary);
		const bool ok = this->write(fout);
		if (!ok)
			std::print("Failed to serialize to {}\n", p.string());
	}
	void read(std::istream& in)
	{
		decltype(signature) test_signature{};
		//TODO
		throw;
	}

	void reset()
	{
		this->weights.clear();
		this->biases.clear();
		this->activators.clear();
	}

	static vector& get_thread_local_helper_vector()
	{
		static thread_local vector v;
		return v;
	}

	void eval(vector& x) const
	{
		vector& y = get_thread_local_helper_vector();

		const size_t n_layers = this->weights.size();
		for (size_t i = 0; i < n_layers; ++i)
		{
			multiply_mat_vec(this->weights[i], x, y);
			add_vec_vec(this->biases[i], y);
			activate_vector(y, this->activators[i]);
			std::swap(x, y);
		}
	}
};

auto create_preset_topology_nn()
{
	const auto layers = { 40u, 10000u, 2u };
	nn_t nn;
	nn.create(layers);
	return nn;
}


int main()
{
	auto nn = create_preset_topology_nn();
	nn.write("a.txt");

	vector x;


	auto clock = std::chrono::steady_clock::now;
	decltype(clock() - clock()) dt{};

	const size_t n = 1000, k = 50;
	for (size_t i = 0; i < n + k; ++i)
	{
		x.resize(65536);
		auto t1 = clock();
		nn.eval(x);
		auto t2 = clock();

		if (i > k)
			dt += t2 - t1;
	}

	std::print("{}", std::chrono::duration_cast<std::chrono::microseconds>(dt / n));


	std::ofstream() << x[1];
}
