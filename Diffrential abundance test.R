library(readxl)
library(car)
library(readr)
library(base)
library(Biobase)
library(ggplot2)
library(RColorBrewer)
library(tidyverse)
library(reshape2)
library(pROC)
library(ggrepel)
library(infotheo)
feat <-read.csv("C:/Users/14529/Desktop/PDACdata/DATA/OTU.csv",header=T,row.names=1)
metaTable <-read.csv("C:/Users/14529/Desktop/PDACdata/PDAC_Cohorts/PDACunmatched.csv",header=T,row.names=1)
featTable = feat[rownames(metaTable), ]
featTable=t(featTable)
metaTable$ID = rownames(metaTable)
p.val <- matrix(NA, nrow=nrow(featTable), ncol=1, 
                dimnames=list(row.names(featTable)))
fc <- p.val
aucs.mat <- p.val
aucs.all  <- vector('list', nrow(featTable))
log.n = 1e-08
for (f in row.names(featTable)) {
    
    
  x <- as.numeric(featTable[f, metaTable %>% 
                                filter(target=="1") %>% pull(ID)])
  y <- as.numeric(featTable[f, metaTable %>% 
                                filter(target=="0") %>% pull(ID)])
    
    # Wilcoxon
  p.val[f,1] <- wilcox.test(x, y, exact=FALSE)$p.value
    
    # AUC
  aucs.all[[f]][[1]]  <- c(roc(controls=y, cases=x, 
                                 direction='<', ci=TRUE, auc=TRUE)$ci)
  aucs.mat[f,1] <- c(roc(controls=y, cases=x, 
                           direction='<', ci=TRUE, auc=TRUE)$ci)[2]
    
    # FC
  q.p <- quantile(log10(x+log.n), probs=seq(.1, .9, .05), na.rm=TRUE) 
  q.n <- quantile(log10(y+log.n), probs=seq(.1, .9, .05), na.rm=TRUE)
  fc[f,1] <- sum(q.p - q.n)/length(q.p)
}
  
p.adj1<-data.frame(p.val)
colnames(p.adj1) <- "adj"
  # add fc and auc
fc <- fc[rownames(p.adj1),]
p.adj1$p.val <- p.val
p.adj1$fc <- fc
  # log2 of fc
p.adj1$log2fc[p.adj1$fc==0] = log2(1)
  #If number < 0 then take log2 of absolute value 
  # and assign the negative number to it
p.adj1$log2fc[p.adj1$fc< 0] = -log2(abs(p.adj1$fc[p.adj1$fc <0]))
  #If number > 0 then take log2 of the number
p.adj1$log2fc[p.adj1$fc> 0] = log2(p.adj1$fc[p.adj1$fc >0])
p.adj1$auc.mat <- aucs.mat
p.adj1$log10p = -log10(as.numeric(p.adj1$adj))
p.adj1=as.data.frame(p.adj1)
p.adj1$species <-  rownames(p.adj1)
p.adj1$significant <- ifelse(p.adj1$adj < 0.01,  "p < 0.01", "not sig")

################################################################################
## Plot wilcoxon test results
filename=p.adj1
# plot the differentially abundant functions
    p <- ggplot(filename, aes(x = fc, y = log10p)) +
      geom_hline(aes(yintercept=2), colour="#BB0000",linetype="dashed", size=0.8)+
      geom_vline(aes(xintercept=0), colour="#BB0000",linetype="dashed",size=0.8)+
      geom_point(aes(color = significant), alpha = 1, size = 4) +
      scale_color_manual(values = c("black","purple")) +
      theme_bw(base_size = 14) +
      geom_text_repel(data = subset(filename, adj < 0.01),
                      aes(label = species),
                      size = 4,
                      box.padding = unit(0.35, "lines"),
                      point.padding = unit(0.5, "lines")) +
      ggtitle("Differential abundance testing (unmatched)") +
      theme(plot.title = element_text(hjust = 0.5)) +
      xlab("log2 fold change") +
      ylab("log10 p-value")
    myplot<-p+theme(panel.grid.major=element_blank(),panel.grid.minor=element_blank())+ylim(0,4.5)+xlim(-4,4)
    myplot
    ggsave(myplot, dpi = 800, filename=paste0("C:/Users/14529/Desktop/",
                              "unmatched1.pdf"),
           width = 8, height=10)
  
