# GroCH - MOA Implementation

This repository contains the implementation of GroCH, iGNGSVM and RCARF, and scripts to run parameter-tuning for the MOA implementation GroCH in an HPC cluster using SLURM.

It also contains the ADDM drift detector, which consists of two embedded ADWIN confidence intervals under the same detector. One to signal warnings and another for concept drifts.

Older versions of GroCH as referred as 'NA'/'NA.java' in the scripts.



## Dependencies: 

In order to the algorithms contained by the folder `src`, the following versions of Java and Massive Online Analysis (MOA) are needed.

- [MOA version 2017.06](https://sourceforge.net/projects/moa-datastream/files/MOA/2017%20June/moa-release-2017.06b.zip/download) (use the developer version with src code: [moa-2017.06-sources](https://repo1.maven.org/maven2/nz/ac/waikato/cms/moa/moa/2017.06/))
- Java SE 7.

The folder `experiments` contains bash scripts to run a similar parameter optimisation and experiments than the ones performed in the PhD thesis **Adaptive Algorithms For Classification On High-Frequency Data Streams: Application To Finance**. 

These scripts have been tested in both the Windows Linux Subsystem ([WSL](https://docs.microsoft.com/en-us/windows/wsl/install)) in Windows 10 and in the MacOS Catalina terminal using the bash shell. WSL might need of a X11 server to show the graphical interface. [Xming](https://sourceforge.net/projects/xming/) and [XLaunch](https://x.cygwin.com/docs/xlaunch/) are two valid tools for this purpose.

The scripts with suffix `_slurm` have been created to trigger experiments in parallel in a SLURM. The version used is:

- SLURM 19.05.1-2



## Setup Instructions

After downloading MOA and installing Java SE 7, to run the algorithms provided in this repository, uncompress [moa-2017.06-sources](https://repo1.maven.org/maven2/nz/ac/waikato/cms/moa/moa/2017.06/) and move the algorithms contained in the folder `src` of this repository inside. The next step is to compile the `src` files to have relevant `.class` files for all. This can be done by mounting the repository in an IDE such as [Eclipse](https://www.eclipse.org/downloads/). These algorithms have been developed using Eclipse IDE 2018-03. Once the .class files are available, the algorithms of this repository should appear when running MOA through the graphical user interface (GUI) or bash. For this purpose, do not run the moa JAR file since this is a self-contained setup.

It is recommended to start using MOA's graphical interface, which will show the different parameters supported by the algorithms and different options available to experiment. This can be run with the code below:

```bash
MOA_DEV="."  
EXTRAPATH="./lib/*":"./src/moa/classifiers"
java -Xmx1024m -cp $EXTRAPATH:$MOA_DEV -javaagent:sizeofag-1.0.0.jar moa.gui.GUI
```

For documentation about the use of MOA, [visit this link](https://moa.cms.waikato.ac.nz/documentation/). Data is expected to be fed in ARFF format or as a data stream. Different tools like [WEKA](https://www.wikihow.tech/Convert-CSV-to-ARFF), or libraries like [csv2arff](https://pypi.org/project/csv2arff/) can be used to convert tabular data (e.g. CSV files) to this format. 

Experiments in MOA can also be run through bash. We have used bash  in this thesis to automate experiments and run them in parallel. 

An example bash script, see `experiments/run_tests_spy2021.sh`. This script receives four arguments:

1. Algorithm or experiment name (depending on the test) (e.g. DWM, CPF or NA (== GroCH)).
2. Sub-experiment name (depending on the first argument, this may second argument may be needed)
3. Random seed (for non-deterministic algorithms or to identify a different dataset)
4. Base classifier (e.g. NB or HT)

For an in-detail descriptions of the parameters of these algorithms, check the PhD thesis **Adaptive Algorithms For Classification On High-Frequency Data Streams: Application To Finance**. Appendix A, in this thesis, lists all the paremeters of the MOA code of GroCH. It also lists important parameters and values for this algorithm and its competitors.

A simpler script to run one of the algorithms of this thesis can be seen below. In the script above, the dataset is read from the file `data.arff` in the folder `data`. The results are outputted to a `log` folder.

```bash
MOA_DEV="."

java -cp "$MOA_DEV/lib/*":$MOA_DEV/moa:$MOA_DEV/weka -javaagent:$MOA_DEV/sizeofag-1.0.0.jar moa.DoTask \
"EvaluateInterleavedTestThenTrain \
 -l (igngsvm.IGNGSVM -t 1500 -e 0-Q 0.0025 -n 3 -T 1 -X 0
 -b (weka.classifiers.functions.LibSVM -S 0 -K 0 -D 1 -G 0.0 -R 0.0 -N 0.5 -M 40.0 -C 8.0 -E 0.0010 -P 0.1 -model $MOA_DEV)) \
 -s (ArffFileStream -f $MOA_DEV/data/data.arff) \
 -e (BasicClassificationPerformanceEvaluator) \
 -f 1500" > $MOA_DEV/logs/test.txt
```

