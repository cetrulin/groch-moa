package moa.classifiers.igngsvm;

import java.lang.reflect.Field;
import java.util.ArrayList;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.igngsvm.gng.GUnit;
import moa.classifiers.igngsvm.gng.GNG;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import moa.options.WEKAClassOption;
import weka.classifiers.*;
import weka.classifiers.functions.LibSVM;
// import weka.core.Instance;
import weka.core.Instances; // TODO: We may need to update this function also, but this can require to rewrite the algorithm
import com.yahoo.labs.samoa.instances.Instance;
import weka.filters.Filter;
import weka.filters.supervised.instance.SMOTE;
import weka.filters.supervised.instance.WilsonEditing;

public class IGNGSVM extends AbstractClassifier implements MultiClassClassifier{
	  	
	private static final long serialVersionUID = 1L;

	/**
	 * @author Andres Leon Suarez Cetrulo
	 * @info  incremental Growing Neural Gas Support Vector Machine
	 **/	
	
	//Parametro bloque IGNGSVM
	public IntOption tsOption = new IntOption("blocksSize", 't',"BlockSize", 10100);
	public IntOption modoTsOption = new IntOption("blocksMode", 'T',"BlockMode", 0);
	
	//Parametros GNG
	public IntOption lambdaOption = new IntOption("lambda", 'l', "Lambda", 100);
	public IntOption maxAgeOption = new IntOption("maxAge", 'm',"MaximumAge", 200);
	public FloatOption alfaOption = new FloatOption("alfa", 'a',"Alfa", 0.5);
	public FloatOption constantOption = new FloatOption("d", 'd',"d", 0.995);
	public FloatOption BepsilonOption = new FloatOption("epsilonB", 'Z',"EpsilonB", 0.2);//antes e
	public FloatOption NepsilonOption = new FloatOption("epsilonN", 'K',"EpsilonN", 0.006);//antes n
	public IntOption stoppingCriteriaOption = new IntOption("stoppingCriteria", 'c', "Stopping criteria", 0);
	public FloatOption perParadaOption = new FloatOption("modoParada", 'Q', "Modo de parada", 20); //20->%20 POR CIENTO DEL TAMANYO DE BLOQUE TS -> si = 0, usa critParada de arriba
	public IntOption modoHerenciaOption = new IntOption("HerenciaSVs", 'e', "herenciaSVs", 0); // 0->herencia igngsvm1, 1->herencia igngsvm2, >1-> sin herencia de SVs de iteraciones pasadas
    public IntOption smoteNNOption = new IntOption("VecinosEnSMOTE", 'X', "NNinSMOTE", 5); // 0->no hace SMOTE
	public FlagOption classAsAttributeOption = new FlagOption("classAsFeature", 'b',
			"Should the class be considered as a feature in the topology?");
	
	//Entrada a parametros LibSVM
	public WEKAClassOption baseLearnerOption = new WEKAClassOption("baseLearner", 'b',
            "Classifier to train.", weka.classifiers.Classifier.class, "weka.classifiers.functions.LibSVM");
	
    //Parametros filtro Wilson
	public IntOption m_kNNOption = new IntOption("KvecinosWilson", 'n', "neighbours", 3); 
	public IntOption vecesWilsonOption = new IntOption("VecesWilson", 'V', "vecesWilson", 1); //Wilson Default 1 vez, es el numero de iteraciones que hace Wilson en cada bloque de ejemplos entrante
	public IntOption wilsonModeOption = new IntOption("ModoWilson", 'M', "modoWilson", 0); //if = 0 -> modo normal, if = 1 -> modo solo SVs
	
	//Objeto SVM
	public LibSVM svm;

	//Prototype set
	private ArrayList<ArrayList<GUnit>> S;
	
	//Support Vector set
	public ArrayList<GUnit> SV;
				
	//nle
    protected int numberInstances;
    
    //Bloque de datos TS
    protected Instances TS;
    
    //Lista de bloques de datos con distintas etiquetas
	public ArrayList<Instances> TSi;
            
    protected Instances instancesBuffer;

    protected boolean isClassificationEnabled;

    protected boolean isBufferStoring;
    
    protected int valorBufferTS;
    
    protected boolean trainTS = false;
    
	public boolean isRandomizable() {
		return false;
	}	
	
	boolean m_debug = false;
	boolean m_exportS = false;

	boolean m_export = true;
	
	// Preparing for learning
	public void resetLearningImpl() {
		SV = new ArrayList<GUnit>();
		S = new ArrayList<ArrayList<GUnit>>();
		
        try {
            String[] options = weka.core.Utils.splitOptions(baseLearnerOption.getValueAsCLIString());
            createWekaClassifier(options);
        } catch (Exception e) {
            System.err.println("Creating a new classifier: " + e.getMessage());
        }
        
        numberInstances = 0;
        isClassificationEnabled = false;
        this.isBufferStoring = true;
	}
	
    @Override
    public void trainOnInstanceImpl(Instance inst) {
        
    		//Variables auxiliares para el metodo
		double [][] SVpnew = null;
		int []label, nSV, claseSV; 
		int anterior;
		
		//Instancias de otras clases auxiliares
		Instances Temp;
		
		//Instances GNG list
		ArrayList<GNG> GNGi = new ArrayList<GNG>();
		ArrayList<Double> labels = new ArrayList<Double>();
		ArrayList<Instances> Splus = new ArrayList<Instances>();;
		
		try {
            if (numberInstances == 0) {
                this.instancesBuffer = new Instances (((weka.core.Instance)(inst.copy())).dataset());
                this.instancesBuffer.clear();
                this.valorBufferTS =inst.index(0);
                
                if (svm instanceof UpdateableClassifier) {
                    svm.buildClassifier(instancesBuffer);
                    this.isClassificationEnabled = true;
                } else {
                    this.isBufferStoring = true;
                }
                
            }numberInstances++;

            if (svm instanceof UpdateableClassifier) {
                if (numberInstances > 0) {
                    ((UpdateableClassifier) svm).updateClassifier((weka.core.Instance) inst);
                }
            } else {

                 if (isBufferStoring == true) {
                     instancesBuffer.add((weka.core.Instance) inst);
                     if(modoTsOption.getValue()!=0 && (instancesBuffer.get(instancesBuffer.numInstances()-1).value(0)!=valorBufferTS)){
                    	 valorBufferTS=(int)instancesBuffer.get(instancesBuffer.numInstances()-1).value(0);
                    	 if(instancesBuffer.size()>1)
                    	 	trainTS = true;
                     } 
                 }
                 
                 ///////////////////////////// TS segun valor primer atributo
                 if(modoTsOption.getValue()==0 && (numberInstances % tsOption.getValue() == 0 && instancesBuffer.size() == tsOption.getValue())){
                	 	trainTS = true;
                 } if(trainTS){
                	 	trainTS=false;            	 
			        //Construimos TS con las instancias del buffer
				    	TS = new Instances(instancesBuffer);
				    	TSi = new ArrayList<Instances>();
			        isBufferStoring = false;
					S.clear();                	 
					//System.out.println("El numero de vecinos en Wilson es: "+m_kNNOption.getValue());
					//System.out.println("Clasificador IGNGSVM"+(modoHerenciaOption.getValue()+1));
								
					// Se divide el bloque TS en tantos bloques TSi como numero i de clases
					//Recorremos el bloque de instancias para separar estas en subconjuntos dependiendo de su clase
	                	for (int i = 0; i < TS.numInstances(); i++) {
	    					Instance aux = (Instance) TS.get(i).copy();
	    					double auxLabel = aux.classValue();
	                		    					
	                		//Si la clase ya existe se agrega a su subconjunto de datos
	                		if(labels.contains(auxLabel)){
	        					Instances TsiAux = TSi.get(labels.indexOf(auxLabel));
	                			TsiAux.add((weka.core.Instance) aux);
	                			TSi.set(labels.indexOf(auxLabel),TsiAux);
	                			
	                    	//Si no existe se agrega a la lista de subconjuntos
	                		} else {
	        					Instances TsiAux = new Instances (((weka.core.Instance)(inst.copy())).dataset());
	        					TsiAux.clear();
	                			TsiAux.add((weka.core.Instance) aux);
	                			TSi.add(TsiAux);
	                			labels.add(auxLabel);
	                		}                		
					}
                	
	                	// MODO 2 HERENCIA DE SVs
					Instances SVs;
					SVs = new Instances (((weka.core.Instance) inst).dataset());
					SVs.clear();

					if (modoHerenciaOption.getValue() == 1) {
		    	        		for (int i = 0; i < SV.size(); i++) {
		    	        			Instance instSV = (Instance) inst.copy();
		    	        			for (int j = 0; j < SV.get(i).w.length; j++)
		    	        				instSV.setValue(j,SV.get(i).w[j]);
		    	        			
		    						instSV.setClassValue(SV.get(i).getLabel());
		    						SVs.add((weka.core.Instance) instSV);
		    	        		}
	    	        		
    					// Se dividen el bloque de SVs en tantos bloques SVi como numero i de clases
    					// Recorremos el bloque de instancias para separar estas en subconjuntos dependiendo de su clase
                    	for (int i = 0; i < SVs.numInstances(); i++) {
        					Instance aux = (Instance) SVs.get(i).copy();
        					double auxLabel = aux.classValue();
                    		    					
                    		//Si la clase ya existe se agrega a su subconjunto de datos
                    		if(labels.contains(auxLabel)){
            					Instances TsiAux = TSi.get(labels.indexOf(auxLabel));
                    			TsiAux.add((weka.core.Instance) aux);
                    			TSi.set(labels.indexOf(auxLabel),TsiAux);
                    			
                        	//Si no existe se agrega a la lista de subconjuntos
                    		} else {
            					Instances TsiAux = new Instances (((weka.core.Instance)(inst.copy())).dataset());
            					TsiAux.clear();
                    			TsiAux.add((weka.core.Instance) aux);
                    			TSi.add(TsiAux);
                    			labels.add(auxLabel);
                    		}
    					}
                	}// FIN MODO 2 HERENCIA DE SVs
					
                	//INICIO GNG
                	// Una vez separados los datos en sub-conjuntos se crean i objetos GNG distintos que se entrenaran 
                	// con sus bloques de datos correspondientes dependiendo de la clase       		                
					trainTopologies(GNGi); 
	        		printSubsetsSi(labels);
	        		
	        		// System.out.println();
	        		// System.out.println("Inicio bloque TS - Se han procesado sus subconjuntos en instancias de GNG separadas");
            		// System.out.println();
	        		Temp = new Instances (((weka.core.Instance) inst).dataset());
	        		SVpnew = null;

	        		// Combining all subsets Si into S*
				int contador = 0;	
	        		for (int i = 0; i < S.size(); i++) {
	        			Temp.clear();
	        			
		        		//System.out.println("Numero de instancias de la topologia GNG"+ i +": "+S.get(i).size());
		        		for (int k = 0; k < S.get(i).size(); k++, contador++) {
			        		Instance instS = (Instance) inst.copy();
			        		
			        		// Obtaining reference vectors W
		        			for (int j = 0; j < S.get(i).get(k).w.length; j++) {
				        		instS.setValue(j,S.get(i).get(k).w[j]);
				        		
		        			} //Se extienden los vectores de referencia W con la etiqueta del subgrupo GNGi
						instS.setClassValue(labels.get(i));
						Temp.add((weka.core.Instance) instS);		        			
		        		}Splus.add(new Instances (Temp));
	        		}
	        		//END OF GNG
	        		
                	//INICIO SMOTE
                	if(smoteNNOption.getValue()!=0){
	                	int greatestLabel = 0;
	                	for (int i = 0; i < Splus.size(); i++)
	                		if(Splus.get(i).numInstances()>greatestLabel) // Se fija la clase con mayor numero de Instancias
	                			greatestLabel= Splus.get(i).numInstances();
	                	for (int i = 0; i < Splus.size(); i++) {
	                		if(Splus.get(i).numInstances()!=greatestLabel){
			        			System.out.println("TAMANYO ANTES DE SMOTE: "+Splus.get(i).numInstances());
			                	SMOTE filters =new SMOTE(); 
			                	filters.setInputFormat(Splus.get(i));
			                	filters.setNearestNeighbors(smoteNNOption.getValue()); // El numero de vecinos por defecto es 5.
			        			filters.setPercentage((double)(greatestLabel-Splus.get(i).numInstances())/(double)Splus.get(i).numInstances()*100);
			        			Splus.set(i, Filter.useFilter(Splus.get(i), filters));
			        			System.out.println("TAMANYO DESPUES DE SMOTE: "+Splus.get(i).numInstances());
	                		}
	                	}
                	} //FIN SMOTE
                	
                	Temp.clear();
                	
                	//Unimos Splus en un solo objeto de instancias (Temp)
                	for (int i = 0; i < Splus.size(); i++) 
						for (int j = 0; j < Splus.get(i).size(); j++) 
							Temp.add(Splus.get(i).get(j));                	

	        		//MODO 1 DE HERENCIA DE SVs: Se combina S* con los SV de la pasada iteracion
                	if (modoHerenciaOption.getValue() == 0) {
    					SVs = new Instances (Temp); 
    					SVs.clear();
		        		for (int i = contador; i < SV.size()+contador; i++) {
		        			Instance instSV = (Instance) inst.copy();
		        			for (int j = 0; j < SV.get(i-contador).w.length; j++) {
			        			instSV.setValue(j,SV.get(i-contador).w[j]);
			        		}
		        			instSV.setClassValue(SV.get(i-contador).getLabel());
							if (wilsonModeOption.getValue() == 0)
								Temp.add((weka.core.Instance) instSV); //se anyaden a la lista de prototipos sobre la que pasar Wilson
							else
								SVs.add((weka.core.Instance) instSV); //se anyaden a lista de SVs sobre la que pasar Wilson mode 2
		        		} 
                	}// FIN MODO 1 DE HERENCIA DE SVs 
                	
                	//INICIO WILSON EDITING
					for (int vezWilson = 0; vezWilson < vecesWilsonOption.getValue(); vezWilson++) {						
	        			System.out.println("TAMANYO ANTES DE WILSON it. "+vezWilson+":  "+Temp.numInstances());
	                	WilsonEditing filterW = new WilsonEditing(); 
	                    filterW.setInputFormat(Temp);
	        			filterW.setNumberOfNearestNeighbors(m_kNNOption.getValue());
	        			filterW.setWilsonMode(wilsonModeOption.getValue());	        			
	        			if(wilsonModeOption.getValue()==1)//if wilsonMode is equal to subpopulation, it sends SVs
	            			filterW.setWilsonSubPopulation(SVs);
	                    Temp = Filter.useFilter(Temp, filterW);
	        			System.out.println("TAMANYO DESPUES DE WILSON it. "+vezWilson+": "+Temp.numInstances());
					}//FIN WILSON EDITING				

					for (int i = 0; i < Temp.numInstances(); i++) {
						
						System.out.println(Temp.get(i));
					}
					
	        		try {
	        			//Train a new SVM with the set Temp
	        			buildClassifier(Temp);
	        			System.out.println("Crea de forma satisfactoria el clasificador GNGSVM");
	        			System.out.println("Longitud instancesBuffer "+instancesBuffer.numInstances());	        
                        isClassificationEnabled = true;
                        instancesBuffer = new Instances (((weka.core.Instance)(inst.copy())).dataset());
                        instancesBuffer.clear();
                        isBufferStoring = true;

	        		} catch (Exception e) {
	        			e.printStackTrace();
	        		} 

	        		//Get the new support vector set SVpnew       	    				        			
        			label =  (int[]) getField (svm.m_Model,"label");
        			nSV =  ((int[]) getField (svm.m_Model,"nSV"));        			
        			Object [][]o =  (Object[][]) getField (svm.m_Model,"SV");
        			SVpnew = new double[o.length][];       			
        			claseSV = new int [SVpnew.length]; 
        			
        			for (int i = 0; i < o.length; i++) {
        				SVpnew [i]= new double [o[i].length]; 
						for (int j = 0; j < o[i].length; j++)
							SVpnew[i][j] =((Double) getField (o[i][j],"value"));
						anterior = 0;
						for(int j = 0;j<nSV.length;){
							if (i<nSV[j]+anterior){
								claseSV[i]=label[j];
								j = nSV.length;
							}else {
								anterior+=nSV[j];
								j++;
							}
						}
						
					}//Update old support vector set SV
	        		SV.clear();
	        		for(int i = 0;i<SVpnew.length;i++)
        				SV.add(new GUnit(SVpnew[i],claseSV[i]));
	        			
	        		if(m_export){
		        		System.out.println();
		        		System.out.println("LISTA DE SV: ");
		        		System.out.println();
		        		
	        		} for (int i = 0; i < SV.size(); i++)
		        		System.out.println(SV.get(i)+","+claseSV[i]);
	        		
	        		if(m_export){
		        		System.out.println();
		                System.out.println("Fin bloque TS");
		                System.out.println();
		                System.out.println("//////////////////////////////////////");
	                }
                }	
            } 

        } catch (Exception e) {
            System.err.println("Training: " + e.getMessage());
            e.printStackTrace();
        }
        
    }

	private void printSubsetsSi(ArrayList<Double> labels) {
		if (m_exportS){
			System.out.println();
				System.out.println("LISTA DE S: ");
				System.out.println();
			for (int i = 0; i < S.size(); i++)
				for (int j = 0; j < S.get(i).size(); j++) 
		    		System.out.println(S.get(i).get(j)+", "+labels.get(i));
					
		} // All the i mini-batches in TS have been processed
	}

	private void trainTopologies(ArrayList<GNG> GNGi) {
		for (int i = 0; i < this.TSi.size(); i++) {
			
			// Initialize a GNG object for training
			GNG aux = new GNG(this.lambdaOption, this.alfaOption, this.maxAgeOption,
							 this.constantOption, this.BepsilonOption, this.NepsilonOption, 
							 this.classAsAttributeOption);
			aux.resetLearning();
			
			// Set stopping criteria (fixed or as a percentage)
			if (perParadaOption.getValue()!=0){
				IntOption nParadaTSi = (IntOption) this.stoppingCriteriaOption.copy();
				int percentParada = (int) ((this.perParadaOption.getValue()*(double)TSi.get(i).size())/(double)100);
				if(percentParada>=1){
					nParadaTSi.setValue(percentParada);
			        	aux.stoppingCriteriaOption = nParadaTSi;
			        	if(this.m_debug) System.out.println("GNG on this TSi for  "+nParadaTSi.getValue()+" created neurons.");
				} else {
					aux.stoppingCriteriaOption=this.stoppingCriteriaOption;
				}
			} else {
				aux.stoppingCriteriaOption=this.stoppingCriteriaOption;
			}
			
			// Send examples to GNGi until the stopping criteria is met
			if(this.TSi.get(i).numInstances()==1) {
				aux.trainOnInstanceImpl((Instance) this.TSi.get(i).get(0));
				
			} else if(TSi.get(i).numInstances()>1){	
				// The instances that belong to the class i are added to the object GNGi
				for(int j=0;aux.getNumberOfPrototypesCreated()<aux.stoppingCriteriaOption.getValue();j++){                        	
			        	aux.trainOnInstanceImpl((Instance) this.TSi.get(i).get(j));
			        	if(j+1==this.TSi.get(i).numInstances()) j = -1;
			    }
			}
			
			// Adding GNGi to the list of GNGis and its set S to the list of subsets of S
			GNGi.add(aux);
		    this.S.add(GNGi.get(i).getS()); 
		    
		} // return S;
	} 
    
	//predict class	
    public double[] getVotesForInstance(Instance inst) {
        double[] votes = new double[inst.numClasses()];
        
        if (isClassificationEnabled == false) {
        	//System.out.println("Warning: Clasifica aleatoriamente! a la altura de "+numberInstances+" entrenadas");

            for (int i = 0; i < inst.numClasses(); i++) {
            	votes[i] = 1.0 / inst.numClasses();
            }
        } else {
            try {
                votes = svm.distributionForInstance((weka.core.Instance) inst);
            } catch (Exception e) {
                System.err.println(e.getMessage());
            }
        }
        return votes;
    }
    
    public void buildClassifier(Instances Temp) {
        try {
            if ((svm instanceof UpdateableClassifier) == false) {                       	
                try {
                	svm = new LibSVM();
                    String[] options = weka.core.Utils.splitOptions(baseLearnerOption.getValueAsCLIString());
                    createWekaClassifier(options);

                    System.out.println("ENUMERA INSTANCIAS 6");
	        		System.out.println(Temp.isEmpty());
	        		
	        		System.out.println("ENUMERA INSTANCIAS 7");
	        		System.out.println(Temp.numInstances());
                
                    svm.buildClassifier(Temp);

                } catch (Exception e) {
                    System.err.println("Creating a new classifier: " + e.getMessage());
                }

                isBufferStoring = false;
            }
        } catch (Exception e) {
            System.err.println("Building WEKA Classifier: " + e.getMessage());
        }
    }
	
    public void createWekaClassifier(String[] options) throws Exception {
        String classifierName = options[0];
        String[] newoptions = options.clone();
        newoptions[0] = "";
        this.svm = (LibSVM) weka.classifiers.AbstractClassifier.forName(classifierName, newoptions);
    }
    
    /**
     * returns the current value of the specified field.
     * 
     * @param o           the object the field is member of
     * @param name        the name of the field
     * @return            the value
     */
    protected Object getField(Object o, String name) {
      Field       f;
      Object      result;
      
      try {
        f = o.getClass().getField(name);
        result = f.get(o);
      }
      catch (Exception e) {
        e.printStackTrace();
        result = null;
      }
      
      return result;
    }

    //////////////////////////////////
	
    @Override
    public void getModelDescription(StringBuilder out, int indent) {
         out.append(("iGNGSVM v2 by Andres Leon Suarez Cetrulo. "+svm.toString()));
    }
    
	protected moa.core.Measurement[] getModelMeasurementsImpl() {
		return null;
	}


}
