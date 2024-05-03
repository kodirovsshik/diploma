
export module diploma.image;
import diploma.utility;
import diploma.lin_alg;

import <fstream>;
import <ranges>;
import <bit>;

import <defs.hpp>;



EXPORT_BEGIN

enum class threestate
{
	no = 0,
	yes = 1,
	keep = 2,
};

bool threestate_to_bool(threestate state, bool current)
{
	switch (state)
	{
	case threestate::no: return false;
	case threestate::yes: return true;
	case threestate::keep: return current;
	default: std::unreachable();
	}
}

struct image_settings
{
	threestate colors = threestate::keep;
	threestate alpha = threestate::keep;
};

template<class fp_t = fp>
class image
{
public:
	tensor data;

	void reset() noexcept;

	bool read(cpath filename, image_settings settings = {});
	bool read_bmp(cpath filename, image_settings settings = {});

	void write_qoi(cpath filename, image_settings settings = {});
};

EXPORT_END



template<class fp_t>
void image<fp_t>::reset() noexcept
{
	this->n_planes = 0;
	for (auto& vec : this->planes)
		vec.clear();
}



namespace fs = std::filesystem;
namespace vs = std::ranges::views;

template<class T>
T iabs(T x)
{
	if (x < 0)
		return -x;
	return x;
}


#define assert_throw(cond, msg, ...) { if (!(cond)) throw std::format(L##msg __VA_OPT__(,) __VA_ARGS__); }

template<class fp_t>
bool image<fp_t>::read(cpath filename, image_settings settings)
{
	if (read_bmp(filename, settings))
		return true;
	return false;
}

template<class fp_t>
bool image<fp_t>::read_bmp(cpath filename, image_settings settings)
{
	nodestruct std::vector<char> file_buffer;
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
	auto read_byte = [&] { return (uint8_t)file_buffer[file_ptr++]; };

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
		if (file_header.signature != 0x4D42)
			return false;
		data_offset = file_header.data_offset;
	}

	uint32_t n_planes_in;
	bool top_to_bottom;
	size_t width, height;

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

		width = (size_t)info_header.width;
		height = (size_t)iabs(info_header.height);
		top_to_bottom = info_header.height < 0;

		switch (info_header.bpp)
		{
			//case 8:   n_planes_in = 1; break;
		case 24: n_planes_in = 3; break;
		case 32: n_planes_in = 4; break;
		default:
			assert_throw(false, "Unsupported bpp value {} in {}", info_header.bpp, filename.wstring());
		}
	}

	const bool grayscale_out = !threestate_to_bool(settings.colors, n_planes_in >= 3);
	const bool alpha_out = threestate_to_bool(settings.alpha, n_planes_in >= 4);

	const size_t bgr_to_rgb_mapper[] = { 2,1,0,3 };
	const unsigned n_color_planes_out = grayscale_out ? 1 : n_planes_in;
	const unsigned n_color_planes_in = std::min(n_planes_in, 3u);
	const fp_t grayscale_normalizer = (fp_t)1 / (n_color_planes_in * 255);
	const fp_t channel_normalizer = (fp_t)1 / (n_color_planes_in * 255);

	data.resize_storage({ height, width, n_color_planes_out + (bool)alpha_out });

	file_ptr = data_offset;

	const int filler_bytes = -int(n_planes_in * width) & 3;
	for (size_t i = 0; i < height; ++i)
	{
		const size_t y = top_to_bottom ? i : (height - 1 - i);

		for (size_t x = 0; x < width; ++x)
		{
			uint8_t channels[4]{ 0,0,0,255 };
			for (size_t i = 0; i < n_planes_in; ++i)
				channels[i] = read_byte();

			if (grayscale_out)
				data(y, x, 0) = (channels[0] + channels[1] + channels[2]) / grayscale_normalizer;
			else
			{
				if (n_color_planes_in == 1)
					channels[2] = channels[1] = channels[0];

				for (uint32_t plane = 0; plane < n_color_planes_out; ++plane)
					data(y, x, bgr_to_rgb_mapper[plane]) = channels[plane] / channel_normalizer;

				if (alpha_out)
					data(y, x, n_color_planes_out) = channels[3] / channel_normalizer;
			}
		}

		for (int i = 0; i < filler_bytes; ++i)
			(void)read_byte();
	}
}

template<class fp_t>
void image<fp_t>::write_qoi(cpath filename, image_settings settings)
{
	const size_t n_planes_in = data.dims().depth;
	const size_t n_color_planes_in = n_planes_in < 3 ? 1 : 3;

	const bool grayscale_out = !threestate_to_bool(settings.colors, n_color_planes_in > 1);
	xassert(!grayscale_out, "QOI does not support single-channel images");
	const bool alpha_in = n_color_planes_in != n_planes_in;
	const bool alpha_out = threestate_to_bool(settings.alpha, alpha_in);

	const size_t n_color_planes_out = 3;
	const size_t n_planes_out = n_color_planes_out + alpha_out;

	xassert(data.dims().width <= UINT32_MAX && data.dims().height <= UINT32_MAX, "Image can not fir into QOI format");

	std::ofstream fout(filename, std::ios::out | std::ios::binary);
	auto try_write = [&](const void* data, size_t size) {
		fout.write((const char*)data, size);
		xassert(fout, "Failed to write to {}", (char*)filename.generic_u8string().data());
	};
	auto try_write_obj = [&](const auto& obj) {
		try_write(&obj, sizeof(obj));
	};

	{
		struct
		{
			const char magic[4]{ 'q', 'o', 'i', 'f' };
			uint32_t width;
			uint32_t height;
			uint8_t channels;
			const uint8_t colorspace = 1;
		} qoi_header;
		qoi_header.width = std::byteswap((uint32_t)data.dims().width);
		qoi_header.height = std::byteswap((uint32_t)data.dims().height);
		
		qoi_header.channels = (uint8_t)n_planes_out;
		try_write(&qoi_header, 14);
	}

	using pixel = std::array<uint8_t, 4>;

	//pixel array[64]{};
	//pixel prev_pixel{ 0,0,0,255 };
	pixel curr_pixel{ 0,0,0,255 };

	auto fetch_pixel = [&](size_t y, size_t x){
		for (size_t channel = 0; channel < n_planes_in; ++channel)
			curr_pixel[channel] = uint8_t(data(y, x, channel) * 255 + (fp)0.5);
		
		if (n_color_planes_in == 1)
		{
			if (alpha_in) curr_pixel[3] = curr_pixel[1];
			curr_pixel[2] = curr_pixel[1] = curr_pixel[0];
		}
	};

	//auto pixel_array_idx = [](const pixel (&pixel)) {
	//	uint8_t idx = 0;
	//	idx += pixel[0] * 3;
	//	idx += pixel[1] * 5;
	//	idx += pixel[2] * 7;
	//	idx += pixel[3] * 11;
	//	return idx % 64;
	//};
	//auto array_for = [&](const pixel(&pixel))
	//{
	//	return array[pixel_array_idx(pixel)];
	//};

	//struct _chunk_index_t
	//{
	//	std::ofstream& fout;
	//	decltype(try_write)& try_write;
	//
	//	size_t last_idx = -1, last_run = 0;
	//	void operator()(size_t idx)
	//	{
	//		dassert(idx < 64);
	//
	//		if (idx == last_idx && last_run < 62)
	//			++last_run;
	//		else
	//		{
	//			transfer_stored_run();
	//			last_idx = idx;
	//			last_run = 1;
	//		}
	//	}
	//
	//	void transfer_stored_run()
	//	{
	//		if (last_run < 1)
	//			return;
	//		try_write(&last_idx, 1);
	//		if (last_run == 1)
	//			return;
	//		
	//		const size_t tmp = (last_run - 1) | 0b11000000;
	//		try_write(&tmp, 1);
	//		last_run = 0;
	//		last_idx = -1;
	//	}
	//} chunk_index{ fout, try_write };


	

	for (size_t y = 0; y < data.dims().height; ++y)
	{
		for (size_t x = 0; x < data.dims().width; ++x)
		{
			fetch_pixel(y, x);
			const uint8_t op = 0b11111110 | (uint8_t)alpha_out;
			try_write_obj(op);
			try_write(&curr_pixel, n_planes_out);
		}
	}

	uint64_t end_mark = 1;
	end_mark = std::byteswap(end_mark);
	try_write_obj(end_mark);
}
