
export module diploma.thread_pool;

import <stdint.h>;

import <utility>;
import <functional>;
import <thread>;
import <queue>;
import <mutex>;
import <atomic>;



template<class F, class Tuple, class... Args, std::size_t... I>
constexpr decltype(auto) apply_extra_impl(F&& f, Tuple&& t, std::index_sequence<I...>, Args&& ...args)
{
	return std::invoke(std::forward<F>(f), std::get<I>(std::forward<Tuple>(t))..., std::forward<Args>(args)...);
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



export class thread_pool
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
	size_t size() const noexcept
	{
		return this->threads.size();
	}

	thread_pool(unsigned int n_threads = -1)
	{
		if (n_threads == -1)
			n_threads = std::thread::hardware_concurrency();
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
		auto caller = [func = std::forward<F>(func), args = copy_as_tuple(args_)]
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
	void schedule_split_work(size_t begin, size_t size, F&& func, Ts&& ...args_)
	{
		const size_t n = this->threads.size();
		const uint64_t split_size = size / n;
		const uint64_t leftover_size = size % n;

		uint64_t current_offset = begin;
		for (size_t thread_id = 0; thread_id < this->threads.size(); ++thread_id)
		{
			const uint64_t current_size = split_size + bool(thread_id < leftover_size);
			const uint64_t current_end = current_offset + current_size;
			this->schedule_task(std::forward<F>(func), std::forward<Ts>(args_)..., thread_id, current_offset, current_end);
			current_offset = current_end;
		}
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
		for (size_t thread_id = 0; thread_id < this->threads.size(); ++thread_id)
		{
			const uint64_t current_size = split_size + bool(thread_id < leftover_size);
			const uint64_t current_end = current_offset + current_size;
			this->schedule_ranged_task(current_offset, current_end, std::forward<F>(func), std::forward<Ts>(args_)..., thread_id);
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
