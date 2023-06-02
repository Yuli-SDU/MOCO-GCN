library(mediation)

p_mediate = matrix(NA,ncol = ncol(microbiome11),nrow = ncol(metabolites))
p_direct = matrix(NA,ncol = ncol(microbiome11),nrow = ncol(metabolites))
p_inverse_mediate = matrix(NA,ncol = ncol(microbiome11),nrow = ncol(metabolites))
p_inverse_direct = matrix(NA,ncol = ncol(microbiome11),nrow = ncol(metabolites))


for(i in 1:ncol(microbiome)){
  for(j in 1:ncol(exposome)){
    print(i)
    b<-microbiome[,i]
    c<-expsome[,j]
    data<-cbind(b,a,c)#a represents PDAC labels
    data<-as.data.frame(data)
    colnames(data)=c("X","Y","M")
    model.m=lm(M~X,data)
    model.y=lm(Y~X+M,data)
    summary=summary(mediate(model.m, model.y,treat = "X", mediator = "M", boot = T,sims = 1000))
    p_mediate[j,i] = summary$d.avg.p
    p_direct[j,i] = summary$z.avg.p
    #inverse mediate
    colnames(data)=c("M","Y","X")
    model.m=lm(M~X,data)
    model.y=lm(Y~X+M,data)
    summary=summary(mediate(model.m ,model.y,treat = "X", mediator = "M",boot = T,sims = 1000))
    p_inverse_mediate[j,i]=summary$d.avg.p
    p_inverse_direct[j,i]=summary$z.avg.p
    }
  }

  
ind = which(p_mediate<=1,arr.ind = T)
association = data.frame(exposome = colnames(exposome)[ind[,1]],microbiome = colnames(microbiome)[ind[,2]],pvals_mediate = p_mediate[ind],pvals_inverse_mediate = p_inverse_mediate[ind],pvals_direct = p_direct[ind],pvals_inverse_direct = p_inverse_direct[ind],stringsAsFactors = F)
association$name=paste(association$exposome,association$microbiome,sep = "_with_")

###sankey plot
library(networkD3)
library(tidyverse)
library(ggplot2)
library(ggalluvial)
library(gridExtra)
media = read.csv('sig_mediation.csv',row.names = 1)
#exposome-microbe-PDAC
net=media[which(media$pvals_mediate<0.05&media$pvals_inverse_mediate>0.05),c(1,2,3,4,5,6,7,8)]
net$fre=1

myplot = ggplot(net,aes(axis1 = net$Microbe, axis2 = net$Exposure, axis3 = net$PDAC,y= net$fre))+
  scale_x_discrete(limits = c("Microbe", "Exposure", "PDAC")) +
  geom_alluvium(aes(fill = net$Microbe))+
  geom_stratum(width = 0.33) +theme_bw()+theme(legend.position="none",axis.text = element_text(size = 8),axis.title = element_text(size = 9),panel.grid=element_blank())+geom_text(stat = "stratum",cex=5,aes(label = after_stat(stratum)))





