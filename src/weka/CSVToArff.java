package weka;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;

import java.io.File;

public class CSVToArff {

   public static void main(String[] args) throws Exception {

    // load CSV
    CSVLoader loader = new CSVLoader();
    loader.setSource(new File("/Users/leos/Desktop/SPY-second-level/indicators/SPY_secondlevel_2018-04-24_indicators_ttrain.csv"));
    loader.fieldSeparatorTipText();
    
    Instances data = loader.getDataSet();
    
    // save ARFF
    ArffSaver saver = new ArffSaver();
    saver.setInstances(data);
    saver.setFile(new File("/Users/leos/Desktop/SPY-second-level/indicators/SPY_s_Q1Y17.arff"));
    saver.writeBatch();
    // .arff file will be created in the output location
  }
}
/*
java -Xmx3074m -classpath $MOA_DEV/lib/weka-dev-3.7.12.jar weka.core.converters.CSVLoader "/Users/leos/Desktop/SPY-second-level/processed/SPY_seconds_devset_[2018-04-26].csv" > "/Users/leos/Desktop/SPY-second-level/processed/SPY_s_M12Y16.arff"
java -Xmx3074m -classpath $MOA_DEV/lib/weka-dev-3.7.12.jar weka.core.converters.CSVLoader "/Users/leos/Desktop/SPY-second-level/processed/SPY_seconds_ttrainset_[2018-04-26].csv" > "/Users/leos/Desktop/SPY-second-level/processed/SPY_s_Q1Y17.arff"
*/