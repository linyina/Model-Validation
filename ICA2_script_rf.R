### I. CLASSIFICATION 
## Define the task
wine.classif<- makeClassifTask(id="wine.classif", data=train.wine, target="quality")
classif.test<- makeClassifTask(data=test.wine, target="quality")

# Have a look
wine.classif
str(getTaskData(wine.classif))

# Normalize the data
wine.classif <- normalizeFeatures(wine.classif,method = "standardize")
classif.test <- normalizeFeatures(classif.test,method = "standardize")

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



## (1) Random Forest - ensemble of trees
lrn.rf<- makeLearner("classif.randomForest",par.vals = list(ntree = 200, mtry = 3))
lrn.rf$par.set
lrn.rf$predict.type

# Set parameters
rf.par <- makeParamSet(
  makeIntegerParam("ntree",lower = 10, upper = 500),
  makeIntegerParam("mtry", lower = 3, upper = 10),
  makeIntegerParam("nodesize", lower = 10, upper = 50)
)

# Random search for 50 iterations
ranctrl <- makeTuneControlRandom(maxit = 50L)

# 3 fold cross validation
cv3<- makeResampleDesc("CV",iters = 3L)

# Create tune learner
lrn.rf.tuned<- makeTuneWrapper(learner=lrn.rf, resampling=cv3, measures=mmce, 
                                par.set=rf.par,control=ranctrl)
lrn.rf.tuned$id<- "rf.tuned"

mod <- mlr::train(learner=lrn.rf.tuned, task=wine.classif)



## (2) SVM
SVM.untuned<- makeLearner("classif.svm", par.vals = list(kernel='polynomial'))
SVM.untuned$par.set
SVM.untuned$predict.type
getParamSet("classif.svm")
svm.par <- makeParamSet(
  makeDiscreteParam("cost", values=c(2^-15, 2^-7, 1, 2^7, 2^15)),
  makeDiscreteParam("gamma", values=c(2^-15, 2^-7, 1, 2^7, 2^15)),
  makeDiscreteParam("kernel",values=c("polynomial", "radial","sigmoid"))
)

# Grid search
ctrl <- makeTuneControlGrid()
# 3 fold cross validation
cv3<- makeResampleDesc("CV",iters = 3L)
lrn.SVM.tuned<- makeTuneWrapper(learner = SVM.untuned, resampling = cv3, measures = mmce,
                            par.set = svm.par, control = ranctrl)


## (3) Neural Network
lrn.nntrain<- makeLearner("classif.nnTrain", par.vals = list(max.number.of.layers=50))
getParamSet('classif.nnTrain')
nntrain.par <- makeParamSet(
  makeIntegerParam("max.number.of.layers", lower=2, upper=100),
  makeNumericParam("batchsize", lower = 1, upper=100)
) # Neural network with multiple middle layers
# Grid search
ctrl <- makeTuneControlGrid()
# 3 fold cross validation
cv3<- makeResampleDesc("CV",iters = 3L)
lrn.nntrain.tuned<- makeTuneWrapper(learner = lrn.nntrain, resampling = cv3, measures = mmce,
                                par.set = nntrain.par, control = ranctrl)


##  Benchmarking

learners<- c(list(lrn.SVM.tuned),list(lrn.rf.tuned), list(lrn.nntrain.tuned))
bmresults<- benchmark(learners, wine.classif, cv3, measures=list(mse,mmce))
bmresults
