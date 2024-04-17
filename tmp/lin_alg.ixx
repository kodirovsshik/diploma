module;

#include "defs.hpp"

#include <vector>



export module diploma.lin_alg;
import diploma.utility;

export
{
	using fpvector = dynarray<fp>;
	using matrix = dynarray<fpvector>;


	void multiply_mat_vec(const matrix& m, const fpvector& v, fpvector& x)
	{
		if (m.size() == 0)
		{
			x.clear();
			return;
		}

		const size_t n1 = m.size();
		const size_t n2 = m[0].size();
		xassert(n2 == v.size(), "incompatible multiplication: {}x{} by {}", n1, n2, v.size());

		x.resize(n1);
		for (size_t i = 0; i < n1; ++i)
		{
			x[i] = 0;
			for (size_t j = 0; j < n2; ++j)
				x[i] += m[i][j] * v[j];
		}
	}
	void add_vec_vec(const fpvector& x, fpvector& y)
	{
		const size_t n1 = x.size();
		const size_t n2 = y.size();
		xassert(n1 == n2, "incompatible vector addition: {} and {}", n1, n2);

		for (size_t i = 0; i < n1; ++i)
			y[i] += x[i];
	}
	void add_mat_mat(const matrix& x, matrix& y)
	{
		const size_t n1 = x.size();
		const size_t n2 = y.size();
		xassert(n1 == n2, "incompatible matrix addition: {} and {} rows", n1, n2);

		for (size_t i = 0; i < n1; ++i)
			add_vec_vec(x[i], y[i]);
	}

	void init_vector(fpvector& arr, size_t n)
	{
		arr.clear();
		arr.resize(n);
	}

	void init_matrix(matrix& mat, size_t rows, size_t columns)
	{
		mat.resize(rows);
		for (auto& row : mat)
			init_vector(row, columns);
	}
}
