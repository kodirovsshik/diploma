
#include <filesystem>
#include <fstream>
#include <limits>
#include <print>

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
	{"Cardiomegaly"},
	{"Hernia"}, //110
	{"Mass"},
	{"Nodule"},
	{"Infiltration"},
	{"Emphysema"}, //892
	{"Effusion"},
	{"Atelectasis"},
	{"Pneumothorax"},
	{"Pleural_Thickening"},
	{"Fibrosis"},
	{"Consolidation"},
	{"Edema"}, //628
	{"Pneumonia"}, //322
};


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

int main1()
{
	cpath dataset_root = "D:/archive(1)";

	auto csv = csvreader::new_from_file(dataset_root / "Data_Entry_2017.csv");
	csv.skip_line();

	auto unique_category = []
	(const auto& line) -> bool
	{
		return !line[1].contains('|');
	};

	cpath images_old = dataset_root / "images";
	cpath images_new = dataset_root / "images_filtered";

	std::filesystem::create_directory(images_new);
	for (size_t i = 0; i < std::size(diseases); ++i)
		std::filesystem::create_directory(images_new / std::to_string(i));

	while (true)
	{
		const auto line = csv.read_line();
		if (line.empty())
			break;

		cpath in_file = images_old / line[0];

		if (!unique_category(line))
			continue;
		if (!std::filesystem::exists(in_file))
			continue;

		const auto cateory_directory = std::to_string(disease_to_idx(line[1]));
		cpath out_file = images_new / cateory_directory / line[0];

		std::filesystem::rename(in_file, out_file);
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
		return main1();
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