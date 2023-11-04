
import std;
import <stdint.h>;
import <math.h>;

struct nn_layer_descriptor
{
	size_t output_image_count;
	size_t kernel_width;
	size_t kernel_height;
	size_t pooling_factor;
#define pooling_factor_w pooling_factor
#define pooling_factor_h pooling_factor
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

template<
	size_t _topology_size,
	std::array<nn_layer_descriptor, _topology_size> topology,
	nn_image_stage_descriptor input_image_descriptor
>
class nn_t
{
public:
	using fp = double;
	//static constexpr auto topology = _topology;

	constexpr static size_t convolution_layers = _topology_size - 1;
	
	template<size_t stage>
	consteval auto calculate_image_stage()
	{
		if (stage > 0 && stage - 1 >= _topology_size)
			std::unreachable();

		nn_image_stage_descriptor image = input_image_descriptor;
		for (size_t i = 0; i < stage; ++i)
		{
			image.width = safe_div(image.width - topology[i].kernel_width + 1, topology[i].pooling_factor_w);
			image.height = safe_div(image.height - topology[i].kernel_height + 1, topology[i].pooling_factor_h);
		}
		if (stage != 0) image.count = topology[stage - 1].output_image_count;
		return image;
	}
};

constexpr auto create_preset_topology_nn()
{
	constexpr nn_image_stage_descriptor input{ 3, 256, 256 };
	constexpr nn_layer_descriptor l0{ 5, 5, 5, 2 };
	constexpr nn_layer_descriptor l1{ 8, 3, 3, 2 };
	constexpr std::array topology{ l0, l1 };
	return nn_t<topology.size(), topology, input>();
}



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
		std::make_index_sequence<std::tuple_size_v<Tuple>>(),
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


int main()
{
	const unsigned n_threads = 4;
	thread_pool pool(n_threads);

	auto payload = [&]
	(size_t i)
	{
			std::this_thread::sleep_for(std::chrono::seconds(1));
			std::print("worker {} done\n", i);
	};

	const unsigned n = 10;
	uint64_t dt = 0, k = 5, k0 = 1;
	for (size_t i = 0; i < k; ++i)
	{
		std::print("i = {}\n", i);
		pool.schedule_sized_work(0, n, payload);

		const auto clock_f = std::chrono::steady_clock::now;
		auto t1 = clock_f();
		pool.barrier();
		auto t2 = clock_f();

		if (i >= k0)
			dt += std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	}
	std::print("{} ms\n", dt / (k - k0));
}
