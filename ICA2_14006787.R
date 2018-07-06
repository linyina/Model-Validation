##################################################################################
#####                This script is for STAT3019 ICA2                        #####
#####  --------------------------------------------------------------------  #####
##### 
library(lattice)
library(mlr)
library(mlbench)
library(caret)
library(party)

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

#########################
### STEP 2: Exploratory Analysis
#########################
summary(wine)
summary(winequality.red)
summary(winequality.white)

### Split the data
wine.data<- wine
ind <- createDataPartition(wine.data$quality, p = 0.8, list = FALSE)
train.wine <- wine.data[ind,]
test.wine <- wine.data[-ind,]
str(train.wine)
str(test.wine)
summary(train.wine)
# Checking the skewness: 
# residual.sugar and free sulfur dioxide's five numbers indicates some skewness
hist(train.wine$residual.sugar, breaks = 300, main = "Residual Sugar Mass Concentration",xlab = "Residual Sugar")
hist(train.wine$free.sulfur.dioxide, breaks=300, main= "Mass Concentration of free sulfur dioxide", xlab = "free sulfur dioxide")

# Checking outliers:  
boxplot(train.wine$residual.sugar,train.wine$fixed.acidity, xaxt='n')
axis(1,at=c(1,2), labels=c("Residual Sugar","Volatile acidity"))
boxplot(train.wine$volatile.acidity, train.wine$citric.acid, train.wine$sulphates,xaxt='n') # Yes
axis(1,at=c(1,2,3), labels=c("Volatile acidity", "Citric acid","Sulphates"))
boxplot(train.wine$free.sulfur.dioxide, train.wine$total.sulfur.dioxide,xaxt='n') # Yes
axis(1,at=c(1,2), labels=c("Free Sulfur Dioxide","Total Sulfur Dioxide"))
boxplot(train.wine$chlorides) #Yes
boxplot(train.wine$pH)

# Checking the difference of wine in different colour
par(mfrow=c(2,2))
boxplot(train.wine$total.sulfur.dioxide ~ train.wine$colour, main="Total Sulfur dioxide") # W
boxplot(train.wine$free.sulfur.dioxide ~ train.wine$colour, main="Free Sulfur Dioxide") # W
boxplot(train.wine$residual.sugar ~ train.wine$colour, main="Residual sugar") # W
boxplot(train.wine$citric.acid ~ train.wine$colour, main="Citric Acid") # W
boxplot(train.wine$fixed.acidity ~ train.wine$colour, main="Fixed Acidity") # R
boxplot(train.wine$volatile.acidity ~ train.wine$colour, main="Volatile Acidity") # R
boxplot(train.wine$pH ~ train.wine$colour, main="pH") #R
boxplot(train.wine$sulphates ~ train.wine$colour, main="Sulphates") # R

### Do some testings
testing1<- function(variable){
  print(t.test(variable~colour, data=train.wine))
  print(var.test(variable~colour, data=train.wine))
  print(pairwise.wilcox.test(variable, colour, data=train.wine))
}

attach(train.wine)
testing1(fixed.acidity)
testing1(volatile.acidity)
testing1(citric.acid)
testing1(quality)
testing1(alcohol) #pairwise wicox: p-value=0.28
testing1(pH)  # F-test: p-value=0.1627
testing1(free.sulfur.dioxide)
testing1(sulphates)
testing1(total.sulfur.dioxide)
testing1(density)
testing1(residual.sugar)
testing1(chlorides)
detach(train.wine)

### Pair plots
pairs(train.wine, col=train.wine$colour)
legend("topright", col=wine$colour,legend =c("red", "white"))

### Density Plot
densityplot(~ fixed.acidity + alcohol + residual.sugar,col=c("blue","green","pink"), data=train.wine, width = 1, 
            key = list(lines = list(col=c("blue","green","pink")),
                       text = list(c("fixed.acidity", "alcohol", "residual.sugar")), x=0.8, y=0.8, corner.x=0,corner.y=0 ))

densityplot(~ volatile.acidity + citric.acid + chlorides + density + sulphates + pH,data=train.wine, width = 1,
            col=c(1,2,3,4,5,6),
            key = list(lines = list(col=c(1,2,3,4,5,6)),
                       text = list(c("volatile.acidity", "citirc.acid", "chlorides","density", "sulphates","pH")), x=0.8, y=0.8, corner.x=0,corner.y=0 ))
densityplot(~ free.sulfur.dioxide + total.sulfur.dioxide,data=train.wine, width = 1, col=c(1,2),
            key = list(lines = list(col=c(1,2)),
                       text = list(c("free.sulfur.dioxide","total.sulfur.dioxide")), x=0.8, y=0.8, corner.x=0,corner.y=0 ))


### Correlaiton- find collinearity
wine.cor<- cor(train.wine[,-13], method="spearman")
wine.cor>=0.7  # Free sulfur dioxide and total sulfur dioxide

prcomp(data.frame(train.wine$free.sulfur.dioxide,train.wine$total.sulfur.dioxide)) # 1PC representing 97% variation
train.sulfur<-prcomp(data.frame(train.wine$free.sulfur.dioxide,train.wine$total.sulfur.dioxide))$x[,1]
# Same operation to the test data
test.sulfur<- prcomp(data.frame(test.wine$free.sulfur.dioxide,test.wine$total.sulfur.dioxide))$x[,1]

scale(train.wine$residual.sugar)
scale(test.wine$residual.sugar)

########################
### STEP 3: MODELLING
########################

### I. CLASSIFICATION 
## Define the task
wine.classif<- makeClassifTask(id="wine.classif", data=train.wine, target="quality")
classif.test<- makeClassifTask(data=test.wine, target="quality")  # For testing use
# And another train set without colour to compare
wine.classif.without.colour<- makeClassifTask(id="wine.classif.no.col", data=train.wine[,-13], target="quality")

# Have a look
wine.classif
str(getTaskData(wine.classif))

# Select Feature
impotance<-generateFilterValuesData(wine.classif, method = c("information.gain","chi.squared"))
plotFilterValues(importance,n.show = 20)
fv <- generateFilterValuesData(wine.classif, method = c("information.gain"))
#to launch its shiny application
plotFilterValuesGGVIS(importance)
### Keep the 50% most important featuers
filtered.task = filterFeatures(wine.classif, fval = fv, perc = 0.5, method = "information.gain")
filtered.task

### Fuse a learner with a filter method
lrn <- makeFilterWrapper(leaner = "classif.svm", fw.method = "information.gain", fw.perc = 0.5)

# Grid search
ctrl <- makeTuneControlGrid()
ranctrl <- makeTuneControlRandom(maxit = 50L)

## Choose the resampling strategy:inner
rdesc <- makeResampleDesc("Holdout")
# 3 fold cross validation: outer
cv3<- makeResampleDesc("CV",iters = 3L)

## (1) My own random forest with 100 trees

lrn.tree<-makeLearner("classif.rpart")
?makeBaggingWrapper
lrn.tree.bagged<- makeBaggingWrapper(
  lrn.tree,
  bw.replace = F, # subsampling rather than bootsampling
  bw.iters = 100,
  bw.feats = 0.5
)
lrn.tree.bagged$par.set

## (2) Random Forest - ensemble of trees
lrn.rf<- makeLearner("classif.randomForest",par.vals = list(ntree = 100, mtry = 3))
lrn.rf$par.set
lrn.rf$predict.type
# Set parameters
rf.par <- makeParamSet(
  makeIntegerParam("ntree",lower = 10, upper=200),
  makeIntegerParam("mtry", lower = 3, upper = 10),
  makeIntegerParam("nodesize", lower = 10, upper = 50)
)

lrn.rf.tuned<- makeTuneWrapper(learner=lrn.rf, resampling=rdesc, measures=mmce, 
                               par.set=rf.par,control=ctrl)
lrn.rf.tuned$id<- "rf.tuned"

## (3) SVM - nonlinear
SVM.untuned<- makeLearner("classif.svm", par.vals = list(kernel='polynomial'))
SVM.untuned$par.set
SVM.untuned$predict.type
getParamSet("classif.svm")
svm.par <- makeParamSet(
  makeDiscreteParam("cost", values=c(10^(-2:3))),
  makeDiscreteParam("gamma", values=2^(-5:5)),
  makeDiscreteParam("kernel",values=c("polynomial", "radial","sigmoid"))
)
lrn.SVM.tuned<- makeTuneWrapper(learner = SVM.untuned, resampling = rdesc, measures = mmce,
                                par.set = svm.par, control = makeTuneControlRandom(maxit=30L))

## (4) Neural Network - multiple layers
lrn.nnet<- makeLearner("classif.h2o.deeplearning", par.vals = list(hidden=100))
getParamSet('classif.h2o.deeplearning')
nnet.par <- makeParamSet(
  makeIntegerVectorParam("hidden", len=5,lower=10, upper=500),
  makeNumericParam("rate", lower = 0, upper=1),
  makeNumericParam('epochs', lower= 1, upper=50)
) # Neural network with multiple middle layers
lrn.nnet.tuned<- makeTuneWrapper(learner = lrn.nnet, resampling = rdesc, measures = mmce,
                                    par.set = nnet.par, control = ranctrl)
#################
## BENCHMARK:
# Learners to be compared
learners<- c(list(lrn.tree.bagged),list(lrn.SVM.tuned),list(lrn.rf.tuned))
listMeasures(wine.classif)

## Define a function that calculates the confidence interval
mmce.se <- function(task, model, pred, feats, extra.args) {
  response = getPredictionResponse(pred)
  truth = getPredictionTruth(pred)
  error = measureMMCE(truth, response)
  1.96 * sqrt((error * (1- error))/nrow(getTaskData(task)))# Wilson score interval
}

## Generate the Measure object
mmce.se = makeMeasure(
  id = "mmce.conf", name = "MMCE conf.int",
  properties = c("classif", "classif.multi", "req.pred", "req.truth"),
  minimize = TRUE, best = 0, worst = 100,
  fun = mmce.se
)

measures<- list(acc,mmce,mmce.se,  timetrain)

bmresults<- benchmark(learners, filtered.task, cv3, measures=measures)
bmresults
bmr2<- benchmark(lrn.nnet.tuned, filtered.task, cv3, measures = measures)
bmr2


### PREDICTION
mod <- mlr::train(learner=??, task=wine.classif)
predict <- mlr::predict(mod, test.wine)



### II. REGRESSION

wine.regr<- makeRegrTask(id="wine.regr", data=train.wine, target="quality")
regr.test<- makeRegrTask(data=test.wine, target="quality")  # For testing use
# And another train set without colour to compare
wine.regr.without.colour<- makeRegrTask(id="wine.regr.no.col", data=train.wine[,-13], target="quality")

### (1) Random forest
reg.rf<- makeLearner("regr.randomForest",par.vals = list(ntree = 100, mtry = 3))
# Set parameters
rf.par <- makeParamSet(
  makeIntegerParam("ntree",lower = 10, upper=200),
  makeIntegerParam("mtry", lower = 3, upper = 10),
  makeIntegerParam("nodesize", lower = 10, upper = 50)
)

reg.rf.tuned<- makeTuneWrapper(learner=reg.rf, resampling=rdesc, measures=mse, 
                               par.set=rf.par,control=ctrl)
reg.rf.tuned$id<- "reg.rf.tuned"


## (2) SVM
reg.svm<- makeLearner("regr.svm", par.vals = list(kernel='radial'))
svm.par <- makeParamSet(
  makeDiscreteParam("cost", values=c(10^(-2:3))),
  makeDiscreteParam("gamma", values=2^(-5:5)),
  makeDiscreteParam("kernel",values=c("polynomial", "radial","sigmoid"))
)
reg.SVM.tuned<- makeTuneWrapper(learner = reg.svm, resampling = rdesc, measures = mse,
                                par.set = svm.par, control = ranctrl)
reg.SVM.tuned$id<- "reg.SVM.tuned"


## (3) Neural Network
reg.nnet<- makeLearner("regr.h2o.deeplearning", par.vals = list(hidden=10))
nnet.par <- makeParamSet(
  makeIntegerVectorParam("hidden", len=5,lower=10, upper=500),
  makeNumericParam("rate", lower = 0, upper=1),
  makeNumericParam('epochs', lower= 1, upper=50)
) # Neural network with multiple middle layers
reg.nnet.tuned<- makeTuneWrapper(learner = reg.nnet, resampling = rdesc, measures = mmce,
                                 par.set = nnet.par, control = ranctrl)

### BENCHMARKING
reg.learners<- c(list(reg.rf.tuned),list(reg.SVM.tuned),list(reg.nnet.tuned))
listMeasures(wine.regr)


mse.se<- function(task, model, pred, feats, extra.args) {
  response = getPredictionResponse(pred)
  truth = getPredictionTruth(pred)
  MSE = measureMSE(truth, response)
  qchisq(p=p,df=n)
  measureMSE +/- const * sqrt( (error * (1 - error)) / n)
}
mse.se <- makeMeasure(
  id = "mse.conf", name = "MSE conf.int",
  properties = c("regr","req.pred","req.truth"),
  minimize = TRUE, best = 0, worst = 1,
  fun = mse.se
)

reg.measures<- list(mse, mse.se, mae, timeboth)

measureMSE
measureRMSE

