##################################################################################
#####                This script is for STAT3019 ICA2                        #####
#####  --------------------------------------------------------------------  #####
##### 
library(lattice)
library(mlr)
library(mlbench)
library(caret)
library(party)
library(ggplot2)
#########################
### STEP 1: Reading and recoding
#########################

winequality.red <- read.csv("~/Downloads/Study/UCL/STAT3019/STAT3019ICA2/winequality/winequality-red.csv", sep=";")
winequality.white <- read.csv("~/Downloads/Study/UCL/STAT3019/STAT3019ICA2/winequality/winequality-white.csv", sep=";")
winequality.red$colour <- "red"
winequality.white$colour<- "white"
wine<- rbind(winequality.red, winequality.white)
wine$colour<- as.factor(wine$colour)
sum(is.na(wine)) # No missing values
str(wine)
# Change the quality into factors
wine$quality<- as.factor(wine$quality)
#########################
### STEP 2: Exploratory Analysis
#########################
summary(wine)
summary(winequality.red)
summary(winequality.white)

### Split the data
wine.data<- wine
ind <- createDataPartition(wine.data$quality, p = 0.7, list = FALSE)
train.wine <- wine.data[ind,]
test.wine <- wine.data[-ind,]
str(train.wine)
str(test.wine)
summary(train.wine)

# Checking the skewness: 
# residual.sugar and free sulfur dioxide's five numbers indicates some skewness
hist(train.wine$residual.sugar, breaks = 300, main = "Residual Sugar Mass Concentration",xlab = "Residual Sugar")
hist(train.wine$free.sulfur.dioxide, breaks=300, main= "Mass Concentration of free sulfur dioxide", xlab = "free sulfur dioxide")

# Checking outliers: fixed acidity, volatile acidity, 
boxplot(train.wine$fixed.acidity)
boxplot(train.wine$volatile.acidity) # Yes
boxplot(train.wine$citric.acid) # Yes
boxplot( train.wine$residual.sugar, train.wine$free.sulfur.dioxide, train.wine$total.sulfur.dioxide) # Yes
boxplot(train.wine$chlorides) #Yes
boxplot(train.wine$pH)
boxplot(train.wine$sulphates)
###    Testings
### Two-sample tests:
var.test(quality~colour, data=wine) # p-value 8.561e-06
t.test(quality~colour, data=wine) # p-value <2.2e-16
pairwise.wilcox.test(wine$quality, wine$colour)
var.test(fixed.acidity~colour, data=wine) # p-value <2.2e-16
t.test(fixed.acidity~colour, data=wine) # p-value <2.2e-16
pairwise.wilcox.test(wine$fixed.acidity, wine$colour)
var.test(volatile.acidity~colour, data=wine)
t.test(volatile.acidity~colour, data=wine)
pairwise.wilcox.test(wine$volatile.acidity, wine$colour)

### One-sample tests:
t.test(wine$fixed.acidity,wine$volatile.acidity,paired=TRUE)
wilcox.test(wine$fixed.acidity,wine$volatile.acidity,paired=TRUE)
t.test(wine$free.sulfur.dioxide,wine$total.sulfur.dioxide,paired=TRUE)
wilcox.test(wine$free.sulfur.dioxide,wine$total.sulfur.dioxide,paired=TRUE)


###    Plots
pairs(winequality.red)
pairs(winequality.white[,-13])
pairs(wine, col=wine$colour)
legend("topright", col=wine$colour,legend =c("red", "white"))
#dev.copy(pdf, "pair_col.pdf")
#dev.off()

densityplot(~ fixed.acidity + alcohol  +residual.sugar,data=wine, width = 1)
densityplot(~ volatile.acidity + citric.acid + chlorides + density + sulphates + pH,data=wine, width = 1)
densityplot(~ free.sulfur.dioxide + total.sulfur.dioxide,data=wine, width = 1)

wine.cor<- cor(wine[,-13], method="spearman")
redwine.cor<- cor(wine[wine$colour=="red",][,-13], method="spearman")
whitewine.cor<- cor(wine[wine$colour=="white",][,-13], method="spearman")
wine.cor>=0.7
redwine.cor>=0.7
whitewine.cor>=0.7

swine<- scale(wine[,-13])
pairs(swine)

wine.data<-data.frame(swine, colour=wine$colour)
str(wine.data)



capLargeValues(train.wine, target = "",cols = c("ApplicantIncome"),threshold = 40000)

########################
###
########################

### (1) CLASSIFICATION 
## Define the task
wine.classif<- makeClassifTask(id="wine.classif", data=train.wine, target="quality")
wine.data<- getTaskData(wine.classif)
summary(wine.data)
## Define the learner
learners <-makeLearners(c("classif.nnTrain", "classif.svm", "classif.randomForest"), 
                        ids=c("NNtrain", "SVM.untuned","ctree","RF")) 
# makeLearners makes a list


for (i in 1:length(learners)){
  cat("Learner number:",i,"\n")
  print(learners[[i]]$id)
  cat("Set of hyperparameters","\n")
  print(learners[[i]]$par.set)# get the set of hyperparameters
  cat("configured hyperparameter settings","\n")
  print(learners[[i]]$par.vals) # Get the configured hyperparameter settings that deviate from the defaults
  cat("Type of prediction", "\n")
  print(learners[[i]]$predict.type) # Get the type of prediction
  cat("\n")
}


getParamSet("classif.nnTrain")
getParamSet("classif.svm")
getParamSet("classif.randomforest")

tunegridparams <- makeParamSet(
  makeDiscreteParam("cost", values=10^(-2:3)),
  makeDiscreteParam("gamma", values=2^(-5:5)),
  makeDiscreteParam("kernel",values=c("polynomial", "radial","sigmoid"))
)  # non-linear SVM

ctrl = makeTuneControlGrid()
SVM.tuned<- makeTuneWrapper(learners[[2]], cv3, mmce, 
                tunegridparams,
                makeTuneControlGrid())
SVM.tuned$id<- "SVM.tuned"
learners<- c(learners, list(SVM.tuned))
measures<- list(mmce)
bmresults<- benchmark(learners, wine.classif, cv3, measures)
bmresults



tunegridparams <- makeParamSet(
  makeIntegerParam("max.number.of.layers", lower=2, upper=100)
  
) # Neural network with multiple middle layers
NNtrain.tuned<- makeTuneWrapper(learners[[1]], cv3, mmce,
                tunegridparams,
                makeTuneControlGrid())
NNtrain.tuned$id<- "NNtrain.tuned"
learners<- c(learners, list(NNtrain.tuned))
measures<- list()
bmresult<- benchmark(learners, wine.classif, cv3, measures)


# Ensemble 
rf_param <- makeParamSet(
  makeIntegerParam("ntree",lower = 10, upper = 500),
  makeIntegerParam("mtry", lower = 3, upper = 10),
  makeIntegerParam("nodesize", lower = 10, upper = 50)
)

## (2) REGRESSION
wine.regr <- makeRegrTask(id = "wine.regr", data = train.wine, target = "quality")
learners <-makeLearners(c("regr.nnet", "regr.svm", "regr.ctree", "regr.randomForest"), 
                        ids=c("Nnet", "SVM.untuned","ctree","RF")) 


