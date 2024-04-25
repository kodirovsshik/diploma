
#include "defs.hpp"

import diploma.nn;
import diploma.lin_alg;

int main()
{
	tensor t({ 2,1,1 });
	t(0) = 1;
	t(1) = -1;
	
	model m(t.dims());

	m.add_layer(leaky_relu_layer{ .negative_side_slope = 0.1f });
	m.predict(t);

}
