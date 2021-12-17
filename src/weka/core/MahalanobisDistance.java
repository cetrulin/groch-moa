package weka.core;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

/**
 *  <!-- globalinfo-start -->
 * Implementing Mahalanobis distance (or similarity) function.<br/>
 * <br/>
 * One object defines not one distance but the data model in which the distances between objects of that data model can be computed.<br/>
 * <br/>
 * For more information, see:<br/>
 * <br/>
 * Wikipedia. Euclidean distance. URL http://en.wikipedia.org/wiki/Mahalanobis_distance.
 * <p/>
 <!-- globalinfo-end -->
 *
 * @author Adapted by Andres L. Suarez-Cetrulo (andres.suarez-cetrulo@ucd.ie)
 * @author: Initial source code adapted from Andrej Gajduk, Researcher at Max Planck Institute of Molecular Physiology
 *  https://github.com/gajduk/mahalanobis-for-weka/blob/master/MahalanobisDistance.java
 * @version $Retrieved on 04/01/2020$
*/

public class MahalanobisDistance  extends NormalizableDistance
implements Cloneable, TechnicalInformationHandler {

	  /**
	   * Constructs an Mahalanobis Distance object, Instances must be still set.
	   */
	  public MahalanobisDistance() {
	    super();
	  }

	  /**
	   * Constructs an Mahalanobis Distance object and automatically initializes the
	   * ranges.
	   * 
	   * @param data 	the instances the distance function should work on
	   */
	  public MahalanobisDistance(Instances data) {
	    super(data);
	  }

	  
	/** for serialization. */
	private static final long serialVersionUID = 626401420517381161L;

	@Override
	public TechnicalInformation getTechnicalInformation() {
	    TechnicalInformation 	result;
	    result = new TechnicalInformation(Type.MISC);
	    result.setValue(Field.AUTHOR, "Github");
	    result.setValue(Field.TITLE, "Mahalanobis distance");
	    result.setValue(Field.URL, "https://github.com/gajduk/mahalanobis-for-weka");
	    return null;
	}

	@Override
	public String globalInfo() {
	    return 
	            "Implementing Mahalanobis distance (or similarity) function.\n\n"
	          + "One object defines not one distance but the data model in which "
	          + "the distances between objects of that data model can be computed.\n\n"
	          + "For more information, see:\n\n"
	          + getTechnicalInformation().toString();
	}

	public double distance(Instances a, Instances b) {
		if ( a.equalHeaders(b) ) {
			System.err.println("Both datasets must have the same headers");
		}
		return distance(getFeaturesAsMatrix(a), getFeaturesAsMatrix(b));
	}

	public double[][] getFeaturesAsMatrix(Instances a) {
		double[][] res = new double[a.numInstances()][a.numAttributes()];
		for ( int i = 0 ; i < res.length ; ++i ) {
			Instance ii = a.instance(i);
			for ( int k = 0 ; k < res[i].length ; ++k ) {
				res[i][k] = ii.value(k);
			}
		}
		return res;
	}

	/**
	 * Calculate the Mahalanobis distance between two matrices of feature values
	 */
	public double distance(double a[][],double b[][]) {
		int total_num_objects = a.length+b.length;
		if ( a[0].length != b[0].length ) {
			System.out.println("Objects must have the same number of attributes.");
			return -1.0;
		}
		int num_features = a[0].length;
		double mean_diff_row_vector[] = new double[num_features];
		double mean_a_row_vector[] = new double[num_features];
		double mean_b_row_vector[] = new double[num_features];
		
		// 1. Get row diff vectors
		mean_a_row_vector = MatrixUtils.meanRowVector(a);
		mean_b_row_vector = MatrixUtils.meanRowVector(b);
		for ( int i = 0 ; i < mean_diff_row_vector.length ; ++i ) {
			mean_diff_row_vector[i] = mean_a_row_vector[i]-mean_b_row_vector[i];
		}
		
		// 2. Get covariances
		double covariance_a[][] = MatrixUtils.calculateCovariance(a);
		double covariance_b[][] = MatrixUtils.calculateCovariance(b);
		MatrixUtils.multiply(covariance_a,a.length/((double)total_num_objects));
		MatrixUtils.multiply(covariance_b,b.length/((double)total_num_objects));
		
		double covariance_total[][] = new double[num_features][num_features];
		for ( int i = 0 ;  i < num_features ; ++i ) {
			for ( int k = 0 ; k < num_features ; ++k ) {
				covariance_total[i][k] = covariance_a[i][k]+covariance_b[i][k];
			}
		}
		
		// 3. Distance using covariances 
		double distance_squared = MatrixUtils.multiply_matrices(MatrixUtils.multiply_matrices(mean_diff_row_vector, MatrixUtils.inverse(covariance_total)), MatrixUtils.transpose(mean_diff_row_vector))[0][0];
		return Math.sqrt(distance_squared);

	}

	/**
	 * calculate the pooled covariance data
	 * covariance1*weight+covariance2*weight
	 */
	public double[][] pool ( double[][] a, double[][] b , double [][]a_covariance , double [][]b_covariance ) {
		double res[][] = new double[a_covariance.length][a_covariance[0].length];
		for ( int i = 0 ; i < a_covariance.length ; ++i ) {
			for ( int k = 0 ; k < a_covariance[0].length ; ++k ) {
				res[i][k] = ((double)a.length)/(a.length+b.length)*a_covariance[i][k];
			}
		}
		for ( int i = 0 ; i < a_covariance.length ; ++i ) {
			for ( int k = 0 ; k < a_covariance[0].length ; ++k ) {
				res[i][k] += ((double)b.length)/(a.length+b.length)*b_covariance[i][k];
			}
		}
		return res;
	}
	
	  /**
	   * Updates the current distance calculated so far with the new difference
	   * between two attributes. The difference between the attributes was 
	   * calculated with the difference(int,double,double) method.
	   * 
	   * @param currDist	the current distance calculated so far
	   * @param diff	the difference between two new attributes
	   * @return		the update distance
	   * @see		#difference(int, double, double)
	   */
	  protected double updateDistance(double currDist, double diff) {
	    double	result; 
//	    result  = currDist;
	    System.out.println("\n\n\n\nWARNING! Mahalanobis distance has been updated, but this function is not meant to be used!! "
	    		+ "Check updateDistance() in MahalanobisDistance.java\n\\n\\n\\n");
//	    result += Math.abs(diff); 
	    return currDist;
	  }
	  
	@Override
	  public String getRevision() {
	    return RevisionUtils.extract("$Revision: 0000 $");
	  }

//	public static void main(String[] args) {
//		double a[][] = { { 2,2 },
//				{ 2,5},
//				{ 6,5},
//				{ 7,3},
//				{ 4,7},
//				{ 6,4},
//				{ 5,3},
//				{ 4,6},
//				{ 2,5},
//				{ 1,3} };
//		double b[][] = { { 6,5 },
//				{ 7,4},
//				{ 8,7},
//				{ 5,6},
//				{ 5,4} };
//		MahalanobisDistance m = new MahalanobisDistance();
//		System.out.println(" A = ");
//		MatrixUtils.print(a);
//		System.out.println();
//		System.out.println(" B = ");
//		MatrixUtils.print(b);
//		System.out.println();
//		System.out.printf("Mahalanobis distance between matrices A nd B is %.3f",m.distance(a,b));
//	}

}