
export module diploma.data_generator;

import diploma.lin_alg;
import diploma.utility;
import diploma.dataset;

import <random>;

import <defs.hpp>;


EXPORT_BEGIN

std::mt19937_64 datagen_rng(rng_seed);

const size_t shapes_tensor_size = 100;

struct vec2
{
	fp x, y;
};
struct mat2x2
{
	fp data[2][2];
};
vec2 operator*(const mat2x2& m, const vec2& v)
{
	return { v.x * m.data[0][0] + v.y * m.data[0][1], v.x * m.data[1][0] + v.y * m.data[1][1] };
}
vec2 operator+(const vec2& a, const vec2& b)
{
	return { a.x + b.x, a.y + b.y };
}
vec2 operator-(const vec2& a, const vec2& b)
{
	return { a.x - b.x, a.y - b.y };
}

template<class T> 
T random(const T& from, const T& to)
{
	dassert(from <= to);
	if constexpr (std::is_floating_point_v<T>)
		return (T)random<uint64_t>(0, UINT64_MAX) / UINT64_MAX * (to - from) + from;
	else if constexpr (std::is_integral_v<T> && std::is_unsigned_v<T>)
	{
		using ui = uintmax_t;
		ui result = datagen_rng();
		ui range = (ui)to - (ui)from + 1;
		if (range) result %= range;
		result += from;
		return (T)result;
	}
	else if constexpr (std::is_integral_v<T> && std::is_signed_v<T>)
	{
		using uT = std::make_unsigned_t<T>;
		return T(random<uT>(0, uT(to - from)) + from);
	}
	else
	{
		[] <bool t = false>{ static_assert(t); }();
	}
}

void _tensor_draw_line_x(tensor& canvas, vec2 from, vec2 to, size_t image)
{
	if (from.x > to.x) std::swap(from, to);
	const fp x0 = floor(from.x); const fp x1 = ceil(to.x);
	size_t n = size_t(x1 - x0 + 1.5f);
	const fp dy = (to.y - from.y) / n;

	size_t x = (size_t)x0;
	fp y = from.y;

	while (n-- > 0)
	{
		fp _, y_t = modf(y, &_);
		const size_t yl = (size_t)y, yh = (size_t)(y + 1);

		canvas(yl, x, image) = 1 - y_t;
		canvas(yh, x, image) = y_t;

		x += 1;
		y += dy;
	}
}
void _tensor_draw_line_y(tensor& canvas, vec2 from, vec2 to, size_t image)
{
	if (from.y > to.y) std::swap(from, to);
	const fp y0 = floor(from.y); const fp y1 = ceil(to.y);
	size_t n = size_t(y1 - y0 + 1.5f);
	const fp dx = (to.x - from.x) / n;

	size_t y = (size_t)y0;
	fp x = from.x;

	while (n-- > 0)
	{
		fp _, x_t = modf(x, &_);
		const size_t xl = (size_t)x, xh = (size_t)(x + 1);

		canvas(y, xl, image) = 1 - x_t;
		canvas(y, xh, image) = x_t;

		y += 1;
		x += dx;
	}
}
void tensor_draw_line(tensor& canvas, vec2 from, vec2 to, size_t image = 0)
{
	if (abs(to.y - from.y) > abs(to.x - from.x))
		_tensor_draw_line_y(canvas, from, to, image);
	else
		_tensor_draw_line_x(canvas, from, to, image);
}
tensor gen_circle_image_tensor()
{
	tensor canvas{ shapes_tensor_size, shapes_tensor_size };

	const fp full_size = (fp)shapes_tensor_size;
	const fp half_size = full_size / 2;

	const auto ry = random<fp>(full_size / 10, full_size / 3);
	const auto rx = random<fp>(full_size / 10, full_size / 3);

	const auto t0 = random<fp>(0, 3.14159265358979324f);
	const mat2x2 rot{ cos(t0), sin(t0), -sin(t0), cos(t0) };
	
	auto center_coord = [&] { return random<fp>(full_size / 2.9f, full_size * 1.9f / 3); };
	const vec2 center{ center_coord() , center_coord()};

	const fp dt = std::sqrt(2 / (rx * rx + ry * ry));
	const fp cosdt = cos(dt), sindt = sin(dt);
	fp cost = 1, sint = 0;

	tensor out(shapes_tensor_size, shapes_tensor_size);
	for (fp t = 0; t < 2 * 3.14159265358979f; t += dt)
	{
		const fp cos_next = cost * cosdt - sint * sindt;
		const fp sin_next = sint * cosdt + cost * sindt;

		const vec2 from{ rx * cost, ry * sint };
		const vec2 to{ rx * cos_next, ry * sin_next };

		tensor_draw_line(canvas, center + rot * from, center + rot * to);

		cost = cos_next;
		sint = sin_next;
	}

	return canvas;
}
tensor gen_square_image_tensor()
{
	tensor canvas{ shapes_tensor_size, shapes_tensor_size };
	const fp half_size = (fp)shapes_tensor_size / 2;

	const auto t0 = random<fp>(0, 3.14159265358979324f);
	const mat2x2 rot{ cos(t0), sin(t0), -sin(t0), cos(t0) };

	auto coord = [&] { return random<fp>(half_size * 0.1f, half_size * 0.9f); };
	const vec2 center = { half_size, half_size };

	vec2 a{ coord(), 0 }; a = rot * a + center;
	vec2 b{ 0, coord() }; b = rot * b + center;
	vec2 c{ -coord(), 0 }; c = rot * c + center;
	vec2 d{ 0, -coord() }; d = rot * d + center;

	tensor_draw_line(canvas, a, b);
	tensor_draw_line(canvas, b, c);
	tensor_draw_line(canvas, c, d);
	tensor_draw_line(canvas, d, a);

	return canvas;
}
dataset_pair gen_data_pair_circle_square()
{
	static int a = 0, b = 0;
	if (random(0, 1))
	{
		++a;
		return { gen_square_image_tensor(), tensor::from_range({0, 1}) };
	}
	else
	{
		++b;
		return { gen_circle_image_tensor(), tensor::from_range({1, 0}) };
	}
}

dataset_pair gen_data_pair_max1()
{
	static thread_local std::uniform_real_distribution<fp> dist(-10, 10);

	const fp x = dist(datagen_rng);

	const auto in = tensor::from_range({ x });
	const auto label = tensor::from_range({ std::max<fp>(x, 1) });
	return { in, label };
}
dataset_pair gen_data_pair_sum()
{
	static thread_local std::uniform_real_distribution<fp> dist(0, 10);

	const fp x = dist(datagen_rng);
	const fp y = dist(datagen_rng);

	const auto in = tensor::from_range({ x, y });
	const auto label = tensor::from_range({ x + y });
	return { in, label };
}

class stub_dataset
{
public:
	std::vector<dataset_pair> data;

	template<class pair_generator_t>
	stub_dataset(pair_generator_t&& generator, size_t size)
	{
		data.reserve(size);
		while (size-- > 0)
			data.emplace_back(generator());
	}
};

EXPORT_END
