#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <iostream>
#include <vector>
#include <functional>
#include <cmath>
#include <random>

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
	int rows() { return data.size(); }
	int cols() { return data[0].size(); }

	// Print the matrix
	static void print(NNMatrix m) {
		for (int i = 0; i < m.rows(); i++) {
			for (int j = 0; j < m.cols(); j++) {
				std::cout << m.data[i][j] << " ";
			}
			std::cout << "\n";
		}
	}
	// Return a column matrix given a flattened vector
	static NNMatrix fromVector(std::vector<double> vec) {
		NNMatrix res(vec.size(), 1);
		for (int i = 0; i < vec.size(); i++) {
			res.data[i][0] = vec[i];
		}
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
	// Dot product of two matrices
	static NNMatrix dot(NNMatrix a, NNMatrix b) {
		NNMatrix result(a.rows(), b.cols());
		for (int i = 0; i < a.rows(); i++) {
			for (int j = 0; j < b.cols(); j++) {
				for (int k = 0; k < b.rows(); k++) {
					result.data[i][j] += a.data[i][k] * b.data[k][j];
				}
			}
		}
		return result;
	}
	// Element-wise addition
	NNMatrix operator+(NNMatrix b) {
		NNMatrix res = *this;
		res.forEach([b](double *val, int i, int j) {
			*val += b.data[i][j];
		});
		return res;
	}
	// Element-wise multiplication
	NNMatrix operator*(NNMatrix b) {
		NNMatrix res = *this;
		res.forEach([b](double *val, int i, int j) {
			*val *= b.data[i][j];
		});
		return res;
	}
	// Transpose the matrix (Switch rows and columns)
	NNMatrix transpose() {
		NNMatrix res(cols(), rows());
		for (int i = 0; i < rows(); i++) {
			for (int j = 0; j < cols(); j++) {
				res.data[j][i] = data[i][j];
			}
		}
		return res;
	}
};

#endif
