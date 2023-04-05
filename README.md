# MOCO-GCN
MOCO-GCN consists of two main components: a Two-view Co-training Graph Convolutional Networks (GCNs) module that learns different omics data features by distilling knowledge from each other and a View Correlation Discovery Network (VCDN) module that integrates multi-omics data.

Files

main_MOCO-GCN.py: MOCO-GCN for the prediction of PDAC

models.py: MOCO-GCN model

train_test.py: The training and test functions

utils.py: Supporting functions

Random Forest:  The 25-repeat stratified fourfold cross-validation random forest

Differential abundant test.r: The analysis of differential abundance testing
