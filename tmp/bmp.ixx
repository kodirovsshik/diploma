export module diploma.bmp;

import std;

using namespace std;
using namespace filesystem;
using namespace ranges;
using namespace views;

using xpath = const path&;

export template<class fp_t = float>
class bmp_image
{
public:
	std::array<std::vector<fp_t>, 4> planes;
	unsigned n_planes_in;

	void read(xpath filename, bool grayscale = false);
	void reset() noexcept;
};



#define assert_throw(cond, msg, ...) { if (!(cond)) throw std::format(L##msg, __VA_ARGS__); }

#define try_read(data_ptr, size) \
{ \
	const auto pos = (streamoff)fin.tellg(); \
	fin.read((char*)data_ptr, size); \
	const auto read = fin.gcount(); \
	assert_throw(read == size, "Failed to read {} bytes at {} of {}", size, pos, filename.wstring());\
}
#define try_read_obj(obj) try_read(&obj, sizeof(obj))

template<class T>
T iabs(T x)
{
	if (x < 0)
		return -x;
	return x;
}

template<class fp_t>
void bmp_image<fp_t>::read(xpath filename, bool grayscale)
{
	//this->reset();

	ifstream fin(filename, std::ios::in | std::ios::binary);
	assert_throw(fin.is_open(), "Failed to open file: {}", filename.wstring());

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

	int32_t width, height;
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

	fin.seekg(data_offset, std::ios_base::beg);

	auto planes = std::move(this->planes);
	for (auto& plane : planes | take(n_planes_out))
		plane.resize(pixel_count);
	for (auto& plane : planes | drop(n_planes_out))
		plane.clear();
	//for (uint32_t plane = 0; plane < n_planes_out; ++plane)
		//planes[plane].resize(pixel_count);

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
					val += (uint8_t)fin.get();
				planes[0][pixel] = val / ((fp_t)n_color_planes * 255);

				for (uint32_t plane = n_color_planes; plane < n_planes_in; ++plane)
					(void)fin.get();
			}
			else
			{
				for (uint32_t plane = 0; plane < n_planes_out; ++plane)
					planes[plane][pixel] = fin.get() / (fp_t)255;
			}
		}

		for (int i = 0; i < filler_bytes; ++i)
			(void)fin.get();
	}
	assert_throw(fin, "insufficient pixel data: {} pixels expected in {}", pixel_count, filename.wstring());

	if (n_planes_in >= 3)
		std::swap(planes[0], planes[2]); //bmp stores BGR values, not RGB
	this->planes = std::move(planes);
	this->n_planes_in = n_planes_in;
}

template<class fp_t>
void bmp_image<fp_t>::reset() noexcept
{
	this->n_planes_in = 0;
	for (auto& vec : this->planes)
		vec.clear();
}
