
#include <filesystem>
#include <fstream>
#include <limits>
#include <print>
#include <unordered_set>
#include <random>
#define NOMINMAX
#include <Windows.h>


using path = std::filesystem::path;
using cpath = const path&;
using sw = std::string_view;
using csw = const std::string_view&;

#define STREAMSIZE_MAX (std::numeric_limits<std::streamsize>::max())

#define wprint(fmt, ...) [&](){ auto msg = std::format(L ## fmt __VA_OPT__(,) __VA_ARGS__); fputws(msg.data(), stdout); }()
#define wprintln(fmt, ...) [&](){ wprint(fmt  __VA_OPT__(,) __VA_ARGS__); fputws(L"\n", stdout); }()



cpath dataset_root = "C:/dataset_pneumonia/png32";



template<class T> using nl = std::numeric_limits<T>;
template<class T> static constexpr T nlmax_v = nl<T>::max();
template<class T> static constexpr T nlmin_v = nl<T>::min();

template<typename Out, typename In>
Out int_cast(In x)
{
	bool ok = true;

#ifdef _DEBUG
	if (!nl<Out>::is_signed && x < 0) ok = false;

	auto omin_v = nlmin_v<Out>;
	auto omax_v = nlmax_v<Out>;

	if (!nl<In>::is_signed && nl<Out>::is_signed)
		omin_v = std::max<Out>(omin_v, 0);

	if (x < omin_v) ok = false;
	if (x > omax_v) ok = false;

	if (!ok) throw std::format("int_cast: failed to cast {} from {} to {}", x, typeid(In).name(), typeid(Out).name());
#endif

	return (Out)x;
}





const sw diseases[] = {
	{"No Finding"},
	//{"Infiltration"},
	//{"Atelectasis"},
	//{"Effusion"},
	//{"Nodule"},
	//{"Pneumothorax"},
	//{"Mass"},
	//{"Consolidation"},
	//{"Pleural Thickening"},
	//{"Cardiomegaly"},
	//{"Emphysema"},
	//{"Fibrosis"},
	//{"Edema"},
	{"Pneumonia"},
	//{"Hernia"},
};

constexpr size_t categories_count = std::size(diseases);

size_t disease_to_idx(csw disease)
{
	size_t idx = 0;
	for (auto test : diseases)
	{
		if (test == disease)
			return idx;
		++idx;
	}
	throw std::format("disease_to_idx: Unknown disease: {}", disease);
}





std::vector<std::string> split(const std::string& s, std::string_view delims)
{
	std::vector<std::string> result;
	std::string tmp;

	auto substring_end = [&](bool skip_empty = false)
		{
			if (!skip_empty || tmp.size() != 0)
				result.push_back(std::move(tmp));
			tmp.clear();
		};

	for (auto c : s)
	{
		if (delims.contains(c))
			substring_end();
		else
			tmp += c;
	}

	substring_end(true);
	return result;
}

class csvreader
{
	struct M
	{
		std::unique_ptr<std::istream> source;
	} m;

	csvreader(M&& m_) : m(std::move(m_)) {}


public:
	static csvreader new_from_file(cpath p)
	{
		auto f = std::make_unique<std::ifstream>(p);
		if (!f->is_open())
			throw std::format(L"csvreader::new_from_file: failed to open {}", p.generic_wstring());

		return csvreader{ {std::move(f)} };
	}

	void skip_line()
	{
		m.source->ignore(STREAMSIZE_MAX, '\n');
	}
	std::vector<std::string> read_line()
	{
		std::string line;
		std::getline(*m.source, line);
		return split(line, ",;");
	}
};

void dataset_splitter()
{
	auto csv = csvreader::new_from_file(dataset_root / "Data_Entry_2017.csv");
	csv.skip_line();

	auto unique_category = []
	(const auto& line) -> bool
		{
			return !line[1].contains('|');
		};

	cpath images_old_dir = dataset_root / "images";
	cpath images_filtered_dir = dataset_root / "images_filtered";

	std::filesystem::create_directory(images_filtered_dir);
	for (size_t i = 0; i < std::size(diseases); ++i)
		std::filesystem::create_directory(images_filtered_dir / diseases[i]);

	while (true)
	{
		const auto line = csv.read_line();
		if (line.empty())
			break;

		cpath in_file = images_old_dir / line[0];

		if (!std::filesystem::exists(in_file))
			continue;
		if (!unique_category(line))
			continue;

		cpath out_file = images_filtered_dir / line[1] / line[0];

		std::filesystem::rename(in_file, out_file);
	}
}



unsigned directory_files_count(cpath p)
{
	unsigned result = 0;
	for (const auto& entry : std::filesystem::directory_iterator(p))
		if (entry.is_regular_file())
			++result;
	return result;
}

void move_dir_contents(cpath from, cpath to)
{
	for (const auto& entry : std::filesystem::directory_iterator(from))
		std::filesystem::rename(entry.path(), to / entry.path().filename());
}

void test_dataset_generator()
{
	const double test_dataset_fraction = 0.1;

	cpath images_train_dir = dataset_root / "train";
	cpath images_test_dir = dataset_root / "test";

	if (!std::filesystem::exists(images_train_dir))
	{
		cpath legacy_images_train_dir = dataset_root / "images_filtered";
		if (!std::filesystem::exists(legacy_images_train_dir))
			throw std::exception("no /images_filtered or /train directory found");
		std::filesystem::rename(legacy_images_train_dir, images_train_dir);
	}

	std::filesystem::create_directory(images_test_dir);
	for (const auto& disease : diseases)
	{
		cpath dst_dir = images_test_dir / disease;
		std::filesystem::create_directory(dst_dir);
	}

	std::mt19937_64 rng(std::random_device{}());

	std::unordered_set<unsigned> test_indices;
	for (int category_number = 0; category_number < categories_count; ++category_number)
	{
		std::print("Category {} \"{}\": ", category_number, diseases[category_number]);

		cpath current_train_dir = images_train_dir / diseases[category_number];
		cpath current_test_dir = images_test_dir / diseases[category_number];
		move_dir_contents(current_test_dir, current_train_dir);

		const unsigned n_total_images = directory_files_count(current_train_dir);
		const unsigned n_test_images = (unsigned)(n_total_images * test_dataset_fraction * (1 - DBL_EPSILON) + 1);
		std::print("{} total images, {} test images", n_total_images, n_test_images);


		std::uniform_int_distribution<unsigned> distr(0, n_total_images - 1);
		test_indices.clear();
		while (test_indices.size() != n_test_images)
			test_indices.insert(distr(rng));

		unsigned file_idx = 0;
		for (const auto& entry : std::filesystem::directory_iterator(current_train_dir))
		{
			if (!test_indices.contains(file_idx++))
				continue;
			std::filesystem::rename(entry.path(), current_test_dir / entry.path().filename());
		}
		std::println(" - done");
	}
}




bool string_is_number(csw s)
{
	for (char c : s)
		if (c < '0' || c > '9')
			return false;
	return true;
}
void _renamer_disease_to_idx_worker(cpath index_dir)
{
	for (const auto& entry : std::filesystem::directory_iterator(index_dir))
	{
		const auto& disease = entry.path().filename().string();
		if (string_is_number(disease))
		{
			wprintln("Already a number: {}", entry.path().generic_wstring());
			continue;
		}
		const auto idx = disease_to_idx(disease);
		std::filesystem::rename(entry, index_dir / std::to_string(idx));
	}
}

void renamer_disease_to_idx()
{
	_renamer_disease_to_idx_worker(dataset_root / "train");
	_renamer_disease_to_idx_worker(dataset_root / "test");
}

void _renamer_idx_to_disease_worker(cpath index_dir)
{
	for (const auto& entry : std::filesystem::directory_iterator(index_dir))
	{
		const auto& idx_str = entry.path().filename().string();
		
		size_t idx = -1;
		auto from_chars_result = std::from_chars(idx_str.data(), idx_str.data() + idx_str.size(), idx);
		if (from_chars_result.ec != std::errc{} || from_chars_result.ptr - idx_str.data() != idx_str.size())
		{
			wprintln("Already not a number: {}", entry.path().generic_wstring());
			continue;
		}

		const auto& disease = diseases[idx];
		std::filesystem::rename(entry, index_dir / disease);
	}
}
void renamer_idx_to_disease()
{
	_renamer_idx_to_disease_worker(dataset_root / "train");
	_renamer_idx_to_disease_worker(dataset_root / "test");
}





std::wstring native_narrow_to_wide(csw s)
{
	std::wstring wstr(s.size(), '\0');
	MultiByteToWideChar(CP_ACP, 0, s.data(), int_cast<int>(s.size()), wstr.data(), int_cast<int>(wstr.size()));
	return wstr;
}

int main()
{
	auto loc = setlocale(0, "");

	auto report_explicit_exception_type = [](const char* type)
	{
		std::println("EXCEPTION of type {}", type);
	};
	auto report_rtti_exception_type = [&](const auto& excp)
	{
		report_explicit_exception_type(typeid(excp).name());
	};
	auto exception_base_handler = [&](const auto& excp)
	{
		report_rtti_exception_type(excp);
		std::println("what() = {}", excp.what());
	};

	int return_code = -1;

	try
	{
		renamer_idx_to_disease();
		test_dataset_generator();
		renamer_disease_to_idx();
		return_code = 0;
	}
	catch (const std::filesystem::filesystem_error& e)
	{
		report_rtti_exception_type(e);
		wprintln("what() = {}\npath1() = {}\npath2() = {}\nmessage = {}", 
			native_narrow_to_wide(e.what()),
			e.path1().generic_wstring(),
			e.path2().generic_wstring(),
			native_narrow_to_wide(e.code().message())
		);
	}
	catch (const std::exception& e)
	{
		exception_base_handler(e);
	}
	catch (const std::string& e)
	{
		report_explicit_exception_type("std::string");
		std::println("what() = {}", e);
	}
	catch (const std::wstring& e)
	{
		report_explicit_exception_type("std::wstring");
		wprintln("what() = {}", e);
	}
	catch (...)
	{
		report_explicit_exception_type("(unknown)");
	}

	return return_code;
}
