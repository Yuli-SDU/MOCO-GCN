library(ggm)
library(psych)

pvals = matrix(NA,ncol = ncol(microbiome),nrow = ncol(exposome))
cors = matrix(NA,ncol = ncol(microbiome),nrow = ncol(exposome))
for(i in 1:ncol(microbiome)){
  for(j in 1:ncol(exposome)){
    a=microbiome[is.na(microbiome[,i])==F&is.na(exposome[,j])==F,i]
    b=exposome[is.na(microbiome[,i])==F&is.na(exposome[,j])==F,j]
    if(length(a)>2&length(b)>2){
      cor = cor.test(a,b,method = "spearman")
      pvals[j,i] = cor$p.value
      cors[j,i] = cor$estimate
    }
  }
}
qvals = matrix(p.adjust(pvals,method = "BH"),ncol = ncol(pvals))
colnames(qvals) = colnames(microbiome)
rownames(qvals) = colnames(exposome)
colnames(cors) = colnames(microbiome)
rownames(cors) = colnames(exposome)
ind = which(pvals <=1,arr.ind = T)
association = data.frame(exposome = colnames(exposome)[ind[,1]],microbiome = colnames(microbiome)[ind[,2]],Cor = cors[ind],Pval = pvals[ind],Qval = qvals[ind],stringsAsFactors = F)
association$name=paste(association$exposome,association$microbiome,sep = "_with_")
colnames(association)[1:2]=name
qvals[is.na(qvals)]=1
cors=cors[row.names(qvals),colnames(qvals)]
pvals[pvals<0.05]="."
pvals[qvals<0.05]="!"
pvals[pvals>0.05]=NA
qvals[qvals<0.05]="*"
qvals[qvals>0.05]=NA


