# MOCO-GCN
MOCO-GCN, a framework for classification tasks with exposome and gut microbiome data. The model is mainly composed of a Two-view Co-training Graph Convolutional Networks (GCNs) module for learning microbiome and exposome data features and improve the generalization ability of the GCN through the cooperation among multiple learners, and a View Correlation Discovery Network (VCDN) module for multi-omics data integration.

![](/MOCO-GCN.png)

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


