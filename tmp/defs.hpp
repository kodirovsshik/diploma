
#include <conio.h>

#include <print>
#include <source_location>


#define lambda(input, val) [](input){ return (val); }
#define lambdac(input, val) [=](input){ return (val); }

#define xassert(cond, fmt, ...) [&]{ auto sc = std::source_location::current(); \
	if (!(cond)) \
	{\
		std::print("{}:{}: ASSERTION FAILED:\n", sc.file_name(), sc.line()); \
		std::print(fmt __VA_OPT__(,) __VA_ARGS__); \
		while (_getch() != 27); std::exit(-1); \
	} \
}()
