
#include <conio.h>

#include <print>
#include <source_location>
#include <filesystem>
#include <stacktrace>

#define lambda(input, val) [](input){ return (val); }
#define lambdac(input, val) [=](input){ return (val); }

#define rfassert(cond) { if (!(cond)) return false; }

#define nodestruct static thread_local

#define xassert(cond, fmt, ...) [&]{ auto sc = std::source_location::current(); \
	if (!(cond)) \
	{\
		std::println("\nASSERTION FAILED in {}:{}:", sc.file_name(), sc.line()); \
		std::println(fmt __VA_OPT__(,) __VA_ARGS__); \
		std::println("STACK TRACE:\n{}", std::stacktrace::current()); \
		while (_getch() != 27); std::exit(-1); \
	} \
}()
#define dassert(cond) xassert(cond, "Debug assertion \"" #cond "\" has failed")

#define wprint(fmt, ...) { fputws(std::format(fmt __VA_OPT__(,) __VA_ARGS__).data(), stdout); }

using cpath = const std::filesystem::path&;

using fp = float;

#pragma warning(disable : 4005)
#define image_classes_xlist X(0, pneumonia) X(1, other)

#define EXPORT_BEGIN export{
#define EXPORT_END }

#define DO_DEBUG_CHECKS 1

#define xinvoke(fn, args, ...) std::invoke(std::forward<decltype(fn)>(fn), std::forward<decltype(args)>(args)... __VA_OPT__(,) __VA_ARGS__)
