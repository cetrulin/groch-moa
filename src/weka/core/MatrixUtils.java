package weka.core;

import java.util.Arrays;

/**
 * @author: https://github.com/gajduk/mahalanobis-for-weka/blob/master/MahalanobisDistance.java
 * @version $Retrieved on 04/01/2020$
*/
public class MatrixUtils {
	
	/**
	 * prints a matrix of doubles
	 * @param matrix
	 */
	public static void print( double matrix[][] ) {
		for ( int i = 0 ; i < matrix.length ; ++i ) {
			System.out.println(Arrays.toString(matrix[i]));
		}
	}
	
	/**
	 * Calculates and returns a vector containing all the mean column values
	 * @param a
	 * @return
	 */
	public static double[] meanRowVector(double[][] a) {
		double[] res = new double[a[0].length];
		for ( int i = 0 ; i < a[0].length ; ++i ) {
			res[i] = computeMean(a,i);
		}
		return res;
	}
	
	/**
	 * Calculates the mean value for a single column, i.e. an attribute
	 * @param a - the matrix
	 * @param column_index
	 * @return the mean of all values in the column column_index in the matrix a
	 */
	public static double computeMean(double[][] a, int column_index) {
		double avg = 0;
		for ( int i = 0 ; i < a.length ; ++i  ) {
			avg += a[i][column_index];
		}
		avg /= a.length;
		return avg;
	}
	
	/**
	 * Calculates and return the covariance of a matrix, explanation http://en.wikipedia.org/wiki/Covariance_matrix
	 * @param matrix
	 * @return
	 */
	public static double[][] calculateCovariance(double[][] matrix) {
		double centered_matrix[][] = centerMatrix(meanRowVector(matrix), matrix);
		double transponded_matrix[][] = transponse(centered_matrix);
		double product[][] = multiply_matrices(transponded_matrix,centered_matrix);
		multiply(product,1.0/centered_matrix.length);
		return product;
	}
	
	/**
	 * Multiplies a matrix with a constant
	 * the original matrix is not modified
	 */
	public static void multiply(double[][] m, double c) {
		for ( int i = 0 ; i < m.length ; ++i ) {
			for ( int k = 0 ;  k < m[0].length ; ++k ) {
				m[i][k] *= c;
			}
		}
	}
	
	/**
	 * normalize the values in the matrix by subtracting from every attribute value, the mean value for that attribute
	 * Xi = (Xi-Xavg), for each attribute X, for every value i
	 */
	public static double[][] centerMatrix( double mean_row_vector[] , double matrix[][]) {
		double[][] result = new double[matrix.length][matrix[0].length];
		for ( int i = 0 ; i < matrix.length ; ++i ) {
			for ( int k = 0 ; k < matrix[i].length ; ++k ) {
				result[i][k] = matrix[i][k] - mean_row_vector[k];
			}
		}
		return result;
	}
	
	/**
	 * get the transposed matrix, by changing the row and columns
	 * e.g
	 * 	5 4 2
	 *  3 7 8
	 *  becomes ->
	 *  5 3
	 *  4 7
	 *  2 8
	 */
	public static double[][] transponse(double[][] matrix) {
		double result[][] = new double[matrix[0].length][matrix.length];
		for ( int i = 0 ; i < matrix.length ; ++i ) {
			for ( int k = 0 ; k < matrix[i].length ; ++k  ) {
				result[k][i] = matrix[i][k];
			}
		}
		return result;
	}
	
	/**
	 * Multiplies matrices, explanation http://en.wikipedia.org/wiki/Matrix_multiplication
	 */
	public static double[][] multiply_matrices( double a[][] , double b[][] ) {
		if ( a[0].length != b.length ) {
			System.out.println("Matrices dimensions are bad");
			return null;
		}
		double result[][] = new double[a.length][b[0].length];

		for ( int i = 0 ; i < result.length ; ++i ) {
			for ( int k = 0 ; k < result[0].length ; ++k ) {
				double sum = 0;
				for ( int w = 0 ; w < b.length ; ++w ) {
					sum += a[i][w]*b[w][k];
				}
				result[i][k] = sum;
			}
		}
		return result;
	}
	
	/**
	 * calculates the inverse of a matrix, explanation http://mathworld.wolfram.com/MatrixInverse.html
	 */
	public static double[][] inverse(double[][] matrix) {
		if ( matrix.length != matrix[0].length ) {
			System.out.println("Matrix dimensions are wrong");
			return null;
		}
		double det = determinanta(matrix);
		double cofactor_matrix[][] = cofactors(matrix);
		for ( int i = 0 ; i < cofactor_matrix.length ; ++i ) {
			for ( int k = 0 ; k < cofactor_matrix[i].length ; ++k ) {
				if ( (i+k)%2 != 0 ) {
					cofactor_matrix[i][k] *= -1;
				}
			}
		}
		double [][] res = transponse(cofactor_matrix);
		multiply(res,1.0/det);
		return res;
	}

	/**
	 * Calculates and returns the matrixs detemrinanat
	 * e.g.
	 * 5 3
	 * 4 9
	 * => 5*9 - 4*3 = 45-12 = 33 
	 */
	public static double determinanta(double[][] matrix) {
		if ( matrix.length != matrix[0].length ) {
			System.out.println("Matrix dimensions are wrong");
			return -1;
		}
		if ( matrix.length == 1 ) {
			return matrix[0][0];
		}
		if ( matrix.length == 2 ) {
			return matrix[0][0]*matrix[1][1]-matrix[0][1]*matrix[1][0];
		}
		double res = 0;
		for ( int w = 0 ; w < matrix.length ; ++w ) {
			double product = 1;
			for ( int i = 0 ; i < matrix.length ; ++i ) {
				product *= matrix[i][(i+w)%matrix.length];
			}
			res += product;
		}
		for ( int w = 0 ; w < matrix.length ; ++w ) {
			double product = 1;
			for ( int i = 0 ; i < matrix.length ; ++i ) {
				product *= matrix[i][(matrix.length-1-i-w+matrix.length)%matrix.length];
}
			res -= product;
		}
		return res;
	}
	
	/**
	 * get the cofactor matrix for field [row,col]
	 * e.g.
	 * 1 2 3
	 * 4 5 6
	 * 7 8 9
	 * row=1 , col=1
	 * cofactor matrix == 
	 * 1 3
	 * 7 9
	 * row 1 and column 1 are completely removed
	 */
	public static double[][] transformirajMatrica( double matrix[][] , int row , int column ) {
		double result[][] = new double[matrix.length-1][matrix.length-1];
		int i=0;
		for ( int w = 0 ; w < matrix.length ; ++w ) {
			if ( w != row ) {
				int k = 0;
				for ( int q = 0 ; q < matrix.length ; ++q ) {
					if ( q != column ) {
						result[i][k] = matrix[w][q];
						++k;
					}
				}
				++i;
			}
		}
		return result;
	}
	

	/**
	 * multiplies a vector and a matrix by making the vector a matrix with only one row
	 */
	public static double[][] multiply_matrices(double[] mean_diff_row_vector, double[][] inverse) {
		double mean_diff_row_matrix[][] = new double[1][mean_diff_row_vector.length];
		for ( int i = 0 ; i < mean_diff_row_vector.length ; ++i ) {
			mean_diff_row_matrix[0][i] = mean_diff_row_vector[i];
		}
		return multiply_matrices(mean_diff_row_matrix, inverse);
	}

	/**
	 * multiplies a matrix and a vector by making the vector a matrix with only one row
	 */
	public static double[][] multiply_matrices(double[][] multiply_matrices, double[] mean_diff_row_vector) {
		double mean_diff_row_matrix[][] = new double[1][mean_diff_row_vector.length];
		for ( int i = 0 ; i < mean_diff_row_vector.length ; ++i ) {
			mean_diff_row_matrix[0][i] = mean_diff_row_vector[i];
		}
		return multiply_matrices(multiply_matrices,mean_diff_row_matrix);
	}
	
	/**
	 * returns a matrix containing all the cofactors for the initial matrix
	 */
	public static double[][] cofactors(double[][] matrix) {
		if ( matrix.length != matrix[0].length ) {
			System.out.println("Matrix dimensions are wrong");
			return null;
		}
		double res[][] = new double[matrix.length][matrix.length];
		for ( int i = 0 ; i < matrix.length ; ++i ) {
			for ( int k = 0 ; k < matrix[i].length ; ++k ) {
				res[i][k] = determinanta(transformirajMatrica(matrix,i,k));
			}
		}
		return res;
	}


	/**
	 * Transposes a vector making it a column matrix
	 * @param a
	 * @return
	 */
	public static double[][] transpose(double[] a) {
		double[][] temp = new double[1][a.length];
		for ( int i = 0 ; i < a.length ; ++i ) {
			temp[0][i] = a[i];
		}
		return transponse(temp);
	}

}