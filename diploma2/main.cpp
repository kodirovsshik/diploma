
#include "defs.hpp"

import diploma.nn;
import diploma.lin_alg;
import diploma.utility;

int main()
{
	auto image = tensor::from_range({0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1});
	image.reshape({ 5,5 });

	model m(image.dims());
	//m.add_layer(dense_layer(3));
	//m.add_layer(untied_bias_layer());
	//m.add_layer(leaky_relu_layer(0.1f));
	//m.add_layer(dense_layer(2));
	//m.add_layer(untied_bias_layer());
	//m.add_layer(softmax_layer());
	m.add_layer(convolution_layer{ 3, 3, 2 }); //TODO: test
	//m.add_layer(leaky_relu_layer(0.1f));
	m.finish(sgd_optimizer{ .rate = 0.001f }, mse_loss_function{});

	const auto prediction = m.predict(image);
}
