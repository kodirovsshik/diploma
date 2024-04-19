
#include <filesystem>
#include <fstream>
#include <limits>
#include <print>
#include <unordered_set>
#include <random>

using path = std::filesystem::path;
using cpath = const path&;
using sw = std::string_view;
using csw = const std::string_view&;

#define STREAMSIZE_MAX std::numeric_limits<std::streamsize>::max()

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

	if (x < omin_v) ok = false;
	if (x > omax_v) ok = false;

	if (!ok) throw;
#endif

	return (Out)x;
}


std::vector<std::string> split(const std::string& s, std::string_view delims)
{
	std::vector<std::string> result;
	std::string tmp;

	auto substring_end = [&] (bool skip_empty = false)
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
		if (!f->is_open()) throw;

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






const sw diseases[] = {
	{"No Finding"},
	{"Infiltration"},
	{"Atelectasis"},
	{"Effusion"},
	{"Nodule"},
	{"Pneumothorax"},
	{"Mass"},
	{"Consolidation"},
	{"Pleural Thickening"},
	{"Cardiomegaly"},
	{"Emphysema"},
	{"Fibrosis"},
	{"Edema"},
	{"Pneumonia"},
	{"Hernia"},
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
	throw;
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



int test_dataset_generator()
{
	const double test_dataset_fraction = 0.2;

	cpath images_root = "E:/backup/archive(1)/images";
	cpath images_train = images_root / "train";
	cpath images_test = images_root / "test";

	std::mt19937_64 rng(std::random_device{}());

	std::unordered_set<unsigned> test_indices;
	for (int category_number = 0; category_number < categories_count; ++category_number)
	{
		std::print("Category {} \"{}\": ", category_number, diseases[category_number]);

		cpath current_train_dir = images_train / std::to_string(category_number);
		cpath current_test_dir = images_test / std::to_string(category_number);
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

	return 0;
}

int main()
{
	//auto report_exception_type = [](const auto& excp)
	//{
	//	;
	//	std::println("EXCP type")
	//}

	try
	{
		return test_dataset_generator();
	}
	catch (const std::filesystem::filesystem_error& e)
	{
		auto msg = std::format(L"EXCP type filesystem_error\np1() = {}\np2() = {}\n", e.path1().wstring(), e.path2().wstring());
		std::fputws(msg.data(), stdout);
	}
	catch (...)
	{
		std::println("EXCP type (unknown)");
	}

	return -1;
}