# MOCO-GCN
## Description
MOCO-GCN, a framework for classification tasks with exposome and gut microbiome data. The model is mainly composed of a Two-view Co-training Graph Convolutional Networks (GCNs) module for learning microbiome and exposome data features and improve the generalization ability of the GCN through the cooperation among multiple learners, and a View Correlation Discovery Network (VCDN) module for multi-omics data integration.

![](/MOCO-GCN.png)

## Usage

### Dependencies

The following packages are required for MOCO-GCN.

 - python 3.7+
        - pytorch 1.3.0+cpu
        - scikit-learn == 1.0.2
        - torch-scatter == 1.3.2
 	      - torch-sparse == 0.4.3
        - torch-cluster  == 1.4.5
        
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
		
##Running
      
```
$ python ./main_MOCO-GCN.py -input ./125ASV_6variables
```
### File descriptions
`-input`: A data foler includes `X.csv` and `Y.csv`. `X.csv` is a microbiome and exposome abundance matrix, sample as rows, microbiome and exposome as columns, the last 6 or 23 rows are exposome data. `Y.csv` is pancreatic cancer label.

|SampleID|microbe1|microbe2|...|exposusre1|exposure2|...|
|---|---|---|---|---|---|---|
|host1|0|0.01|...|0|1|...|
|host1|0.05|0.02|...|1|1|...|
|host1|0|0.03|...|1|0|...|
|...|...|

## Others

### Parameter description

-num_epoch: the number of iteration while training the graph convolutional networks;

-lr: the learning rate while training the graph convolutional networks;

-num_class: the number of categories of sample labels;

-num_view: the number of multi-omics data type;

-adj_parameter: the average number of edges retained per node in graph convolutional networks (>1).

## Files

main_MOCO-GCN.py: MOCO-GCN for the prediction of PDAC

models.py: MOCO-GCN model

train_test.py: The training and test functions

utils.py: Supporting functions

Random Forest:  The 25-repeat stratified fourfold cross-validation random forest

Differential abundance test.r: The analysis of differential abundance testing


