import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_curve, auc, accuracy_score, matthews_corrcoef
RANDOM_STATE_RF = 347945
RANDOM_STATE_CV = 124213
otu_df = pd.read_excel("OTU_data.xlsx", index_col = 0)
feature_info = pd.read_csv("feature_info.csv", index_col = 0)
otu_df_full = otu_df
print(otu_df.shape[1])
otu_max_abund =  otu_df.mean(axis = 0)
otu_sig = otu_max_abund[otu_max_abund > 0.0001].index
otu_df = otu_df.loc[:, otu_sig]
print(otu_df.shape[1])
num_iterations = 100
def empiricalPVal(statistic, null_dist):
    count = len([val for val in null_dist if val >= statistic])
    p_val = (count + 1)/float(len(null_dist) + 1)
    return p_val

class modelResults:
    def __init__(self):
        self.tprs = []
        self.aucs = []
        self.importances = []
        self.accuracy = []
        self.matthews = []#马修斯相关系数
        self.shuffled_accuracy = []
        self.shuffled_aucs = []
        self.shuffled_matthews = []
        self.shuffled_tprs = []
        self.mean_fpr = np.linspace(0, 1, 101)

    def getMetrics(self, cohort_n):
        metrics = pd.Series([])
        metrics.loc["n_samples"] = cohort_n
        metrics.loc["auc_mean"] = np.mean(self.aucs)
        metrics.loc["auc_std"] = np.std(self.aucs)
        metrics.loc["auc_median"] = np.median(self.aucs)
        metrics.loc["shuffled_auc_mean"] = np.mean(self.shuffled_aucs)
        metrics.loc["shuffled_auc_std"] = np.std(self.shuffled_aucs)
        metrics.loc["shuffled_auc_median"] = np.median(self.shuffled_aucs)
        metrics.loc["p_val"] = np.mean([empiricalPVal(stat, self.shuffled_aucs) for stat in self.aucs])
        metrics.loc["acc_mean"] = np.mean(self.accuracy)
        metrics.loc["acc_std"] = np.std(self.accuracy)
        metrics.loc["matthews_mean"] = np.mean(self.matthews)
        metrics.loc["matthews_std"] = np.std(self.matthews)
        metrics.loc["shuffled_matthews_std"] = np.std(self.shuffled_matthews)
        metrics.loc["shuffled_matthews_mean"] = np.mean(self.shuffled_matthews)
        metrics.loc["shuffled_accuracy_mean"] = np.mean(self.shuffled_accuracy)
        metrics.loc["shuffled_accuracy_std"] = np.std(self.shuffled_accuracy)
        return metrics

    def getImportances(self, col_names):
        avg_imps = np.stack(self.importances)
        avg_imps = pd.DataFrame(avg_imps, columns = col_names).mean(axis = 0)
        return avg_imps

    def plotROC(self, feature_name, save, title, model):
        plt.figure(1, figsize=(6, 6))
        plt.plot([-0.05, 1.05], [-0.05, 1.05], linestyle=':', lw=2, color='k', alpha=.8)
        ##TEST ROC CURVE
        mean_tpr = np.mean(self.tprs, axis=0)
        mean_tpr[-1] = 1.0
        std_tpr = np.std(self.tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        mean_auc = auc(self.mean_fpr, mean_tpr)
        std_auc = np.std(self.aucs)
        plt.plot(self.mean_fpr, mean_tpr, color='b',label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),lw=2, alpha=0.9)
        plt.fill_between(self.mean_fpr, tprs_lower, tprs_upper, color='b', alpha=.3, label=r'$\pm$ 1 std. dev.')
        ##SHUFFLED ROC CURVE
        mean_tpr = np.mean(self.shuffled_tprs, axis=0)
        mean_tpr[-1] = 1.0
        std_tpr = np.std(self.shuffled_tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        mean_auc = auc(self.mean_fpr, mean_tpr)
        std_auc = np.std(self.shuffled_aucs)
        plt.plot(self.mean_fpr, mean_tpr, color='r',label=r'Mean Shuffled-ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),lw=2, alpha=0.9)
        plt.fill_between(self.mean_fpr, tprs_lower, tprs_upper, color='r', alpha=.3, label=r'$\pm$ 1 std. dev.')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        if save :
            plt.savefig(save_path + "ROCs/" + feature_name + "_" + model + ".png", dpi = 300)
        plt.show()



class VariablesCohortClassification:
    def __init__(self, feature_name, cohort, plot, save, title):
        self.feature_name = feature_name
        self.cohort = cohort
        self.plot = plot
        self.save = save
        self.title = title

    def classifyFeature(self):
        X, y = self.buildDataSubset()
        self.GroupCV(X, y)

    ## Preprocess questionnaire matched-pair cohort for classification
    ##      - taxonomic relative abundance data is log-transformed with a pseudocount of 1
    ##      - abundance data is not normally distributed so this transformation makes it more suitable for classification
    def buildDataSubset(self):
        X = otu_df.loc[self.cohort.index, :].astype(float).values
        print(X.shape[1])
        y = self.cohort["target"].astype(float)
        print(y.value_counts())
        return X, y

    ## 25 iteration 4-fold cross validation
    ##      - pairs must be in consecutive rows, are kept grouped between training and test to maintain this balance
    ## 3 standard machine learning classifiers: random forests, ridge-logistic regression, SVM
    ##      - classifiers chosen for performing well with high-dimensional, low sample data that is noisy, and zero-inflated
    ## Target variable shuffled and model trained over same split of data to assess ability for classifier to find signal in noise
    ## Shuffled performance used to obtain significance non-shuffled standard classifiers
    def GroupCV(self, X, y):
        self.rf = modelResults()
        ##100 iterations Group Shuffle-Split Cross Validation (matched case-control pairs remain stratified)
        cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=25,
                                     random_state=RANDOM_STATE_CV)  # 75/25 training/test split for each iteration
        for fold_num, (train, test) in enumerate(cv.split(X, y)):
            y_shuffled = shuffle(y)
            print(str(fold_num), end=", ")
            ##RANDOM FOREST:
            self.trainModel(X[train], X[test], y[train], y[test], False, self.rf)
            self.trainModel(X[train], X[test], y_shuffled[train], y_shuffled[test], True, self.rf)
        if self.plot:
            self.rf.plotROC(self.feature_name, self.save, self.title, "rf")
        print()
        print()


    ##train one of three classifiers on CV iteration
    ##returns performance metrics, feature importances, saves to classifier object
    def trainModel(self, X_train, X_test, y_train, y_test, shuffle, model_type):
        if model_type == self.rf:
            alg = RandomForestClassifier()
            alg = RandomForestClassifier(n_estimators=512, min_samples_leaf=1, n_jobs=-1, bootstrap=True,
                                         max_samples=0.7, class_weight='balanced', random_state=RANDOM_STATE_RF)
            alg.fit(X_train, y_train)
            imp = alg.feature_importances_
        y_pred = alg.predict_proba(X_test)[:, 1]
        y_pred_class = alg.predict(X_test)
        y_true = y_test
        ##Performance Metrics:
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        matthew = matthews_corrcoef(y_true, y_pred_class)
        acc = accuracy_score(y_true, y_pred_class)
        if shuffle:
            # results for shuffled target variable (corrupted) dataset used as null hypothesis
            model_type.shuffled_tprs.append(np.interp(self.rf.mean_fpr, fpr, tpr))
            model_type.shuffled_tprs[-1][0] = 0.0
            model_type.shuffled_aucs.append(roc_auc)
            model_type.shuffled_matthews.append(matthew)
            model_type.shuffled_accuracy.append(acc)
        else:
            # results of classifier on cohort
            model_type.importances.append(imp)
            model_type.tprs.append(np.interp(model_type.mean_fpr, fpr, tpr))
            model_type.tprs[-1][0] = 0.0
            model_type.aucs.append(roc_auc)
            model_type.accuracy.append(acc)
            model_type.matthews.append(matthew)

        ## Aggregate classification results of all questionnaire variables into single csv file for analysis/comparison


class QuestionnaireResults():
    def __init__(self, iters, col_names, model_name, save_path):
        self.iters = iters
        self.col_names = col_names
        self.model_name = model_name
        self.save_path = save_path
        self.model_results = pd.DataFrame([], columns=metrics)
        self.model_importances = pd.DataFrame([], columns=col_names)
        self.model_aucs = pd.DataFrame([], columns=range(iters))
        self.model_shuffled_aucs = pd.DataFrame([], columns=range(iters))

    def AppendModelRes(self, model_obj, cohort_n, feature_name):
        self.model_results.loc[feature_name, :] = model_obj.getMetrics(cohort_n)
        self.model_importances.loc[feature_name, :] = model_obj.getImportances(self.col_names)
        self.model_aucs.loc[feature_name, :] = model_obj.aucs
        self.model_shuffled_aucs.loc[feature_name, :] = model_obj.shuffled_aucs

    def SaveModelDF(self):
        self.model_results.to_csv(self.save_path + self.model_name + "_results.csv")
        self.model_aucs.to_csv(self.save_path + "AUCs/" + self.model_name + "_aucs.csv")
        self.model_shuffled_aucs.to_csv(self.save_path + "AUCs/" + self.model_name + "_shuffled_aucs.csv")
        self.model_importances.to_csv(self.save_path + "Importances/" + self.model_name + "_importances.csv")


### Plot boxplot of distribution of AUC results from cross-validation for top performing variables
def PlotFeatureBox(model_results, model_aucs, path, model):
    temp = model_results[(model_results["p_val"] <= 1) & (model_results["auc_mean"] >= 0.6)].sort_values(
        "auc_median", ascending=False)
    boxplotdata = model_aucs.loc[temp.index, :].values
    boxplotdata = pd.DataFrame(boxplotdata, index=feature_info.loc[temp.index, "plot_name"]).T
    sns.boxplot(data=boxplotdata, notch=False, showfliers=False, palette="Blues_r", orient="h")
    plt.xlabel("AUC")
    plt.xlim(0.5, 1.0)
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(path + "auc_dists_" + model + ".pdf")
    plt.show()


metrics = ["p_val", "n_samples", "auc_mean", "auc_std", "auc_median", "shuffled_auc_mean", "shuffled_auc_std",
           "shuffled_auc_median", "acc_mean", "acc_std",
           "shuffled_matthews_mean", "shuffled_matthews_std", "matthews_mean", "matthews_std", "shuffled_accuracy_mean",
           "shuffled_accuracy_std"]


def PredPipeline(save_path, dir_path):
    feature_list = os.listdir(dir_path)
    print(feature_list)
    col_names = otu_df.columns
    rf_FR = QuestionnaireResults(num_iterations, col_names, "rf", save_path)
    for feature in feature_list:
        feature_name = feature.split(".")[0]
        print("feature_name:"+feature_name)
        if feature_name == "":
            continue
        if feature_name not in feature_info.index.values:
            print("Skipping")
            continue
        cohort = pd.read_csv(dir_path + feature, index_col=0)
        # cohort.index = cohort["num"]
        cohort_n = len(cohort)
        print(feature_info.loc[feature_name, "plot_name"])
        CohClass = VariablesCohortClassification(feature_name, cohort, True, True, "Classification of " + feature_info.loc[feature_name, "plot_name"])
        CohClass.classifyFeature()
        rf_FR.AppendModelRes(CohClass.rf, cohort_n, feature_name)


    rf_FR.SaveModelDF()
    # Plot performance of binary cohorts (disease, lifestyle, etc.)
    PlotFeatureBox(rf_FR.model_results, rf_FR.model_aucs, save_path, "rf")


dir_path = "/Random Forest"
col_names = otu_df.columns
#excluded Groups
save_path = dir_path + "Results_RF_feature_binary/"
dir_path = dir_path + "Feature_Cohorts/"
feature_list = os.listdir(dir_path)
PredPipeline(save_path, dir_path)

