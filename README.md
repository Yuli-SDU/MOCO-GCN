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
	        wget https://github.com/OSU-BMBL/micah/archive/refs/heads/master.zip


Then, unzip the file and go to the folder.

	unzip master.zip && rm -rf master.zip
	cd ./micah-master
        
```
$ python xxx.py
```

# Files

main_MOCO-GCN.py: MOCO-GCN for the prediction of PDAC

models.py: MOCO-GCN model

train_test.py: The training and test functions

utils.py: Supporting functions

Random Forest:  The 25-repeat stratified fourfold cross-validation random forest

Differential abundance test.r: The analysis of differential abundance testing


