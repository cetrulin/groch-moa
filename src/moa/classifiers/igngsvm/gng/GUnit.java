package moa.classifiers.igngsvm.gng;

import java.util.ArrayList;
import java.util.Arrays;

public class GUnit {
	public double w[];
	private int label;	
	private double errorAccum;
	
	public ArrayList<GEdge> neighbors = new ArrayList<GEdge>();
	
	public GUnit(double w[], int label){
		this.w = w.clone();
		this.label = label;
		this.errorAccum = 0;
	}
	
	public void setError(double error){
		this.errorAccum = error;		
	}
	
	public double getError(){
		return this.errorAccum;
	}
	
	public int getLabel(){
		return this.label;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + this.label;
		result = prime * result + Arrays.hashCode(w);
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		GUnit other = (GUnit) obj;
		if (this.label != other.label)
			return false;
		if (!Arrays.equals(w, other.w))
			return false;
		return true;
	}
	
	public void updateW(double ep, double pattern[]){
		for (int i = 0; i < this.w.length; i++) {
			this.w[i] = this.w[i] + ep * (pattern[i] - this.w[i]);
		}
	}
		
	public static int getNumberOfGPrototypes(ArrayList<GUnit> list){
		int n = 0;
		for (GUnit p: list) n++;
		return n;
	}
	
	public static double dist(double w1[],double w2[]){
		double sum = 0;
		//System.out.println(w1.length+" "+w2.length);  // debug
		for (int i = 0; i < w1.length; i++) {
			sum += Math.pow(w1[i]-w2[i],2);
		}
		return Math.sqrt(sum);
	}
	
	public void addNeighbor(GUnit s2){
		GEdge nueva = new GEdge(this,s2);
		this.neighbors.add(nueva);
		s2.neighbors.add(nueva);
	}
	
	public String toString(){
		return Arrays.toString(this.w)+":"+this.label;
	}
	
	public GEdge searchEdge(GUnit s2){
		for (GEdge c : this.neighbors) {
			if(c.p1.equals(s2)||c.p0.equals(s2)) return c;
		}
		return null;
	}
	
	public void updateAges(){
		for (GEdge c : this.neighbors) {
			c.age++;
		}
	}
	
	public void purgueGEdges(double age){
		for (int i = 0;i<this.neighbors.size();i++){
			GEdge c = this.neighbors.get(i);
			if(c.age>=age){
				this.neighbors.remove(c);
				if(c.p0.equals(this)){
					c.p1.neighbors.remove(c);
				} else {
					c.p0.neighbors.remove(c);
				}
			}
		}
	}
	
	public void removeNeighbor(GUnit p){
		for (int i = 0 ; i<this.neighbors.size();i++) {
			if(this.neighbors.get(i).p0.equals(p) || this.neighbors.get(i).p1.equals(p)) 
				this.neighbors.remove(this.neighbors.get(i));
		}
			
	}
	
	public GUnit[] getNeighborhood(){
		GUnit v[] = new GUnit[this.neighbors.size()];
		for (int i = 0; i < v.length; i++) {
			if(this.neighbors.get(i).p0!=this)	
				v[i] = this.neighbors.get(i).p0;
			else 
				v[i] = this.neighbors.get(i).p1;
			
		}return v;
	}

}
