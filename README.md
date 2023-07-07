# MOCO-GCN


## Description

MOCO-GCN, a framework for classification tasks with exposome and gut microbiome data. The model is mainly composed of a Two-view Co-training Graph Convolutional Networks (GCNs) module for learning microbiome and exposome data features and improve the generalization ability of the GCN through the cooperation among multiple learners, and a View Correlation Discovery Network (VCDN) module for multi-omics data integration.

![](/MOCO-GCN.png)

## Usage


### Dependencies

The following packages are required for MOCO-GCN.

 - python 3.7+
 	- pytorch 1.3.0+cpu
 	- torch-scatter == 1.3.2
 	- torch-sparse == 0.4.3
 	- torch-cluster  == 1.4.5
 	- scikit-learn == 1.0.2     
        
 Please install these dependencies.
 
        conda install pandas
        conda install matplotlib
        conda install scikit-learn
        pip install seaborn
        conda install numpy
        conda install random
   
   
   
 ### Installation
 
The source code of MOCO-GCN is freely available at https://github.com/Yuli-SDU/MOCO-GCN. To install MOCO-GCN, you can download the zip file manually from GitHub, or use the code below in Unix.
   	 
	        cd /your working path/ 
	        wget https://github.com/Yuli-SDU/MOCO-GCN/archive/refs/heads/main.zip


Then, unzip the file and go to the folder.

	        unzip main.zip && rm -rf main.zip
	        cd ./MOCO-GCN-main
		
		
## Running
      
      
```
$ python ./main_MOCO-GCN.py -input ./125ASV_23variables
```


### File descriptions

`-input`: A data foler includes `X.csv` and `Y.csv`. `X.csv` is a microbiome and exposome abundance matrix, sample as rows, microbiome and exposome as columns, the last 23 rows are exposome data. `Y.csv` is pancreatic cancer label. `X.csv` looks like:

|SampleID|microbe1|microbe2|...|exposusre1|exposure2|...|
|---|---|---|---|---|---|---|
|host1|0|0.01|...|0|1|...|
|host1|0.05|0.02|...|1|1|...|
|host1|0|0.03|...|1|0|...|
|...|...|

`Y.csv` looks like:
|SampleID|lable|
|---|---|
|host1|0|
|host1|1|
|host1|0|
|...|...|

Notice that the original microbiome data is available at **./Random Forest/OTU_data.xlsx**, the exposome data is available at **./supplementary table/supplementary table 1.csv**. The feature binary cohorts are available at **./Random Forest/Feature_Cohorts/binary_cohorts**.



## Others

### Parameter description

-num_epoch: the number of iteration while training the graph convolutional networks;

-lr: the learning rate while training the graph convolutional networks;

-num_class: the number of categories of sample labels;

-num_view: the number of multi-omics data type;

-adj_parameter: the average number of edges retained per node in graph convolutional networks (>1).



## About the seed

We use the seed 0 to split the dataset and build the model to make sure that the results are reproducible. The seed can be changed by changing the `seed` variable in the `main_MOCO-GCN.py` scripts.



## Files

main_MOCO-GCN.py: MOCO-GCN for the prediction of PDAC

models.py: MOCO-GCN model

train_test.py: The training and test functions

utils.py: Supporting functions

Random Forest:  The 25-repeat stratified fourfold cross-validation random forest

Differential abundance test.r: The analysis of differential abundance testing

mediation analysis.r: The analysis of mediation analysis

Spearman calculation.r: The calculation of Spearman correlation.

125ASV_23variables: The 125 species and 23 exposures data of Spanish cohort.

DE: The 125 species and 14 exposures data of German cohort.

raw data: The whole microbiome and metadata of Spanish and German cohorts.
