
export module diploma.dataset;

import diploma.lin_alg;
import diploma.image;

import <random>;
import <vector>;

import <stdint.h>;

import "defs.hpp";



EXPORT_BEGIN

struct dataset_pair
{
	tensor input;
	tensor expected;
};

template<class T>
concept dataset_wrapper = requires(std::remove_const_t<T> t, const T ct, size_t sz)
{
	{ t.data[sz] } -> std::same_as<dataset_pair&>;
	{ ct.data[sz] } -> std::same_as<const dataset_pair&>;
	{ ct.data.size() } -> std::same_as<size_t>;
};

size_t size(dataset_wrapper auto const& dataset)
{
	return dataset.data.size();
}

void shuffle(dataset_wrapper auto& dataset, size_t n_first = SIZE_MAX)
{
	if (n_first >= size(dataset))
		return;

	static thread_local std::mt19937_64 shuffle_rng;
	for (size_t i = 0; i < n_first; ++i)
		std::swap(at(dataset, i), at(dataset, i + shuffle_rng() % (dataset.data.size() - i)));
}

auto& at(dataset_wrapper auto& dataset, idx_t n)
{
	xassert(n >= 0 && (size_t)n < size(dataset), "dataset_wrapper::at: invalid index {}", n);
	return dataset.data[n];
}
tensor_dims input_size(dataset_wrapper auto const& dataset)
{
	auto& data = dataset.data;
	xassert(!data.empty(), "dataset_wrapper::input_size: Empty dataset");

	tensor_dims input_dims = data[0].input.dims();
	for (const auto& [input, _] : data)
		xassert(input.dims() == input_dims, "dataset_wrapper::input_size: Inconsistent input size across dataset");

	return input_dims;
}

class folder_labels_split_objects_dataset
{
	using nstr = std::filesystem::path::string_type;

	std::vector<nstr> class_labels;

	void read_labels(cpath root)
	{
		for (const auto& entry : std::filesystem::directory_iterator(root))
		{
			xassert(entry.is_directory(), "dataset root must contain folders with labels as names");
			class_labels.push_back(entry.path().filename().native());
		}
	}
	void read_directory(cpath dir, const tensor& expected)
	{
		image image;
		for (const auto& entry : std::filesystem::directory_iterator(dir))
		{
			xassert(entry.is_regular_file(),
				"dataset::dataset(): Extra object {} in dataset subdirectory {}",
				(char*)entry.path().filename().generic_u8string().data(),
				(char*)dir.generic_u8string().data()
			);
			image.read(entry.path(), { .colors = threestate::no, .alpha = threestate::no });
			data.push_back({ std::move(image.data), expected });
		}
	}

public:
	folder_labels_split_objects_dataset(cpath root)
	{
		read_labels(root);

		tensor expected(class_labels.size());
		for (size_t idx = 0; idx < class_labels.size(); ++idx)
		{
			expected[idx] = 1;
			read_directory(root / class_labels[idx], expected);
			expected[idx] = 0;
		}
	}

	std::vector<dataset_pair> data;
};

class csv_dataset
{
public:
	enum class names_options
	{
		maybe_present, //try parse the first row; if strings found, row discarded
		assume_present, //always discard the first row
		assume_absent, //never discard the first row
	};

	csv_dataset(cpath filename, names_options names = names_options::assume_absent, std::string_view delim = ",;\t", std::string_view newline = "\n")
	{
		assert_throw(names != names_options::maybe_present, "not implemented");
		std::ifstream fin(filename);
		assert_throw(fin.is_open(), "Could not open \"{}\"", filename.native_string());
	
		auto next_is = [&](std::string_view sw)
		{
			return sw.contains(fin.peek());
		};
		auto nextchar = [&]
		{
			return (char)fin.get();
		};
		

	}

	std::vector<dataset_pair> data;
};

static_assert(dataset_wrapper<folder_labels_split_objects_dataset>);

EXPORT_END
