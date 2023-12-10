module;

#include <filesystem>
#include <array>
#include <ranges>
#include <vector>
#include <fstream>

#include "defs.hpp"

export module diploma.bmp;

import diploma.lin_alg;


namespace fs = std::filesystem;
namespace vs = std::ranges::views;

using cpath = const fs::path&;


export template<class fp_t = float>
class bmp_image
{
public:
	std::array<dynarray<fp>, 4> planes;
	unsigned n_planes;
	unsigned width, height;

	void read(cpath filename, bool grayscale = false);
	void reset() noexcept;
};



#define assert_throw(cond, msg, ...) { if (!(cond)) throw std::format(L##msg __VA_OPT__(,) __VA_ARGS__); }

template<class T>
T iabs(T x)
{
	if (x < 0)
		return -x;
	return x;
}

template<class fp_t>
void bmp_image<fp_t>::read(cpath filename, bool grayscale)
{
	nofree std::vector<char> file_buffer;
	size_t file_ptr = 0;

	{
		std::ifstream fin(filename, std::ios::in | std::ios::binary);
		assert_throw(fin.is_open(), "Failed to open file: {}", filename.wstring());

		fin.seekg(0, std::ios::end);
		file_buffer.resize(fin.tellg());
		fin.seekg(0, std::ios::beg);

		fin.read(file_buffer.data(), file_buffer.size());

		assert_throw(fin.gcount() == file_buffer.size(), "Failed to read data from {}", filename.wstring());
	}

	auto try_read_obj = [&]
	(auto& obj)
	{
		memcpy(&obj, file_buffer.data() + file_ptr, sizeof(obj));
		file_ptr += sizeof(obj);
	};
	auto read_byte = [&]{ return (uint8_t)file_buffer[file_ptr++]; };

	uint32_t data_offset;

	{
#pragma pack(push, 1)
		struct
		{
			uint16_t signature;
			char _[8];
			uint32_t data_offset;
		} file_header{};
#pragma pack(pop, 1)
		static_assert(sizeof(file_header) == 14);

		try_read_obj(file_header);
		assert_throw(file_header.signature == 0x4D42, "");
		data_offset = file_header.data_offset;
	}

	uint32_t n_planes_in;
	bool top_to_bottom;

	{
		struct
		{
			uint32_t size;
			int32_t width;
			int32_t height;
			uint16_t planes;
			uint16_t bpp;
			uint32_t compression;
			uint32_t bitmap_size;
			uint32_t ppmh;
			uint32_t ppmv;
			uint32_t palette_colors;
			uint32_t important_colors;
		} info_header;
		static_assert(sizeof(info_header) == 40);
		try_read_obj(info_header);

		assert_throw(info_header.compression == 0, "Unsupported compression method {} in {}", info_header.compression, filename.wstring());
		assert_throw(info_header.width >= 0, "Negative width in {}", filename.wstring());

		width = info_header.width;
		height = iabs(info_header.height);
		top_to_bottom = height < 0;

		switch (info_header.bpp)
		{
		//case 8:   n_planes_in = 1; break;
		case 24: n_planes_in = 3; break;
		case 32: n_planes_in = 4; break;
		default:
			assert_throw(false, "Unsupported bpp value {} in {}", info_header.bpp, filename.wstring());
		}
	}
	const size_t pixel_count = (size_t)width * height;
	const unsigned n_planes_out = grayscale ? 1 : n_planes_in;
	const unsigned n_color_planes = std::min(n_planes_in, 3u);
	const fp_t grayscale_normalizer = (fp_t)1 / (n_color_planes * 255);

	file_ptr = data_offset;

	auto planes = std::move(this->planes);
	for (auto& plane : vs::take(planes, n_planes_out))
		plane.resize(pixel_count);
	for (auto& plane : vs::drop(planes, n_planes_out))
		plane.clear();

	const int filler_bytes = -int(n_planes_in * width) & 3;
	for (size_t i = 0; i < height; ++i)
	{
		const size_t row = top_to_bottom ? i : (height - 1 - i);
		
		for (size_t j = 0; j < width; ++j)
		{
			const size_t pixel = row * width + j;
			
			if (grayscale && n_planes_in != n_planes_out)
			{
				int val = 0;
				for (uint32_t plane = 0; plane < n_color_planes; ++plane)
					val += read_byte();
				planes[0][pixel] = val  * grayscale_normalizer;

				for (uint32_t plane = n_color_planes; plane < n_planes_in; ++plane)
					(void)read_byte();
			}
			else
			{
				for (uint32_t plane = 0; plane < n_planes_out; ++plane)
					planes[plane][pixel] = read_byte() / (fp_t)255;
			}
		}

		for (int i = 0; i < filler_bytes; ++i)
			(void)read_byte();
	}

	if (n_planes_out >= 3)
		std::swap(planes[0], planes[2]); //bmp stores BGR values, not RGB
	this->planes = std::move(planes);
	this->n_planes = n_planes_out;
}

template<class fp_t>
void bmp_image<fp_t>::reset() noexcept
{
	this->n_planes = 0;
	for (auto& vec : this->planes)
		vec.clear();
}
