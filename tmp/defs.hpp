
#include <conio.h>

#include <print>
#include <source_location>
#include <filesystem>

#define lambda(input, val) [](input){ return (val); }
#define lambdac(input, val) [=](input){ return (val); }

#define rfassert(cond) { if (!(cond)) return false; }

#define nofree static thread_local

#define xassert(cond, fmt, ...) [&]{ auto sc = std::source_location::current(); \
	if (!(cond)) \
	{\
		std::print("{}:{}: ASSERTION FAILED:\n", sc.file_name(), sc.line()); \
		std::print(fmt __VA_OPT__(,) __VA_ARGS__); \
		while (_getch() != 27); std::exit(-1); \
	} \
}()

using cpath = const std::filesystem::path&;
