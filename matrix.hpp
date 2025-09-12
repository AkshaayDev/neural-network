#ifndef MATRIX_HPP
#define MATRIX_HPP

#include "./neural-network.hpp"

// Minimal matrix class
class NNMatrix {
public:
	std::vector<std::vector<double>> data;

	// Constructors
	NNMatrix() {}
	NNMatrix(std::vector<std::vector<double>> data) : data(data) {}
	NNMatrix(int rows, int cols) {
		this->resize(rows, cols);
	}

	// Getters
	int rows() const { return data.size(); }
	int cols() const { return data[0].size(); }

	// Helpers
	// Print the matrix
	static void print(NNMatrix m) {
		for (int i = 0; i < m.rows(); i++) {
			for (int j = 0; j < m.cols(); j++) {
				std::cout << m[i][j] << " ";
			}
			std::cout << "\n";
		}
	}
	// Return a column matrix (nx1) given a flattened vector
	static NNMatrix fromVector(std::vector<double> vec) {
		NNMatrix res(vec.size(), 1);
		for (int i = 0; i < vec.size(); i++) {
			res[i][0] = vec[i];
		}
		return res;
	}
	// Return a scalar matrix (1x1) given a single scalar
	static NNMatrix fromScalar(double scalar) {
		NNMatrix res;
		res.data = {{scalar}};
		return res;
	}
	// Resize the number of rows and columns
	void resize(int rows, int cols) {
		data.resize(rows);
		for (int i = 0; i < rows; i++) {
			data[i].resize(cols);
		}
	}
	// Apply a function to each element of the matrix with its value, row and column
	void forEach(const std::function<void(double*, int, int)>& func) {
		for (int i = 0; i < rows(); i++) {
			for (int j = 0; j < cols(); j++) {
				func(&data[i][j], i, j);
			}
		}
	}
	// Fill the matrix with a given value
	void fill(double value) {
		forEach([value](double *val, int i, int j) {
			*val = value;
		});
	}

	// Operations
	// Scalar addition
	NNMatrix operator+(double scalar) {
		NNMatrix res = *this;
		res.forEach([scalar](double *val, int i, int j) {
			*val += scalar;
		});
		return res;
	}
	// Element-wise addition
	NNMatrix operator+(const NNMatrix& other) const {
		NNMatrix res = *this;
		res.forEach([other](double *val, int i, int j) {
			*val += other[i][j];
		});
		return res;
	}
	// Scalar subtraction
	NNMatrix operator-(double scalar) {
		NNMatrix res = *this;
		res.forEach([scalar](double *val, int i, int j) {
			*val -= scalar;
		});
		return res;
	}
	// Element-wise subtraction
	NNMatrix operator-(const NNMatrix& other) const {
		NNMatrix res = *this;
		res.forEach([other](double *val, int i, int j) {
			*val -= other[i][j];
		});
		return res;
	}
	// Scalar multiplication
	NNMatrix operator*(double scalar) {
		NNMatrix res = *this;
		res.forEach([scalar](double *val, int i, int j) {
			*val *= scalar;
		});
		return res;
	}
	// Element-wise multiplication
	NNMatrix operator*(const NNMatrix& other) const {
		NNMatrix res = *this;
		res.forEach([other](double *val, int i, int j) {
			*val *= other[i][j];
		});
		return res;
	}
	// Scalar division 
	NNMatrix operator/(double scalar) {
		NNMatrix res = *this;
		res.forEach([scalar](double *val, int i, int j) {
			*val /= scalar;
		});
		return res;
	}
	// Element-wise division
	NNMatrix operator/(const NNMatrix& other) const {
		NNMatrix res = *this;
		res.forEach([other](double *val, int i, int j) {
			*val /= other[i][j];
		});
		return res;
	}
	// Element-wise scalar exponent
	NNMatrix operator^(double scalar) const {
		NNMatrix res = *this;
		res.forEach([scalar](double *val, int i, int j) {
			*val = std::pow(*val, scalar);
		});
		return res;
	}
	// Access data directly (modifiable)
	std::vector<double>& operator[](int row) {
		return data[row];
	}
	// Access data directly (read-only)
	const std::vector<double>& operator[](int row) const {
		return data[row];
	}
	// Dot product of two matrices
	static NNMatrix dot(const NNMatrix& a, const NNMatrix& b) {
		NNMatrix result(a.rows(), b.cols());
		for (int i = 0; i < a.rows(); i++) {
			for (int j = 0; j < b.cols(); j++) {
				for (int k = 0; k < b.rows(); k++) {
					result[i][j] += a[i][k] * b[k][j];
				}
			}
		}
		return result;
	}
	// Transpose the matrix (Switch rows and columns)
	NNMatrix transpose() const {
		NNMatrix res(cols(), rows());
		for (int i = 0; i < rows(); i++) {
			for (int j = 0; j < cols(); j++) {
				res[j][i] = data[i][j];
			}
		}
		return res;
	}
};

#endif
