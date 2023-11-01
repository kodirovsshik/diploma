
import std;

struct nn_layer_descriptor
{
	size_t output_image_count;
	size_t kernel_width;
	size_t kernel_height;
	size_t pooling_factor;
#define pooling_factor_w pooling_factor
#define pooling_factor_h pooling_factor
};

struct nn_image_stage_descriptor
{
	size_t count;
	size_t width;
	size_t height;
};

template<class T>
constexpr T safe_div(T dividend, T divisor)
{
	if (dividend % divisor) std::unreachable();
	return dividend / divisor;
}

template<
	size_t _topology_size,
	std::array<nn_layer_descriptor, _topology_size> topology,
	nn_image_stage_descriptor input_image_descriptor
>
class nn_t
{
public:
	using fp = double;
	//static constexpr auto topology = _topology;

	constexpr static size_t convolution_layers = _topology_size - 1;
	
	template<size_t stage>
	consteval auto calculate_image_stage()
	{
		if (stage > 0 && stage - 1 >= _topology_size)
			std::unreachable();

		nn_image_stage_descriptor image = input_image_descriptor;
		for (size_t i = 0; i < stage; ++i)
		{
			image.width = safe_div(image.width - topology[i].kernel_width + 1, topology[i].pooling_factor_w);
			image.height = safe_div(image.height - topology[i].kernel_height + 1, topology[i].pooling_factor_h);
		}
		if (stage != 0) image.count = topology[stage - 1].output_image_count;
		return image;
	}
};

constexpr auto create_preset_topology_nn()
{
	constexpr nn_image_stage_descriptor input{ 3, 256, 256 };
	constexpr nn_layer_descriptor l0{ 5, 5, 5, 2 };
	constexpr nn_layer_descriptor l1{ 8, 3, 3, 2 };
	constexpr std::array topology{ l0, l1 };
	return nn_t<topology.size(), topology, input>();
}

int main()
{
	auto nn = create_preset_topology_nn();
	nn_image_stage_descriptor img;

	img = nn.calculate_image_stage<0>();
	img = nn.calculate_image_stage<1>();
	img = nn.calculate_image_stage<2>();
}
