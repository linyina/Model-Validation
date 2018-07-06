library(mlr)

treelearner<-makeLearner("classif.ctree", id = "ctree")

help("makeLearner")

sonar.data<- Sonar
sonar.task<- makeclassiftask(id="sonar", data=sonar.data, target="class")
rm(Sonar)
sonar.task$type
getTaskData(sonar)
sonar.pca<-princomp(sonar.data[,1:60])

sonar.kpcs<- kpca(~., data=sonar.data[,1:60])


# Grid-tuning a support vector machine on the sonar dataset
lrn.SVM <- makelearner("classif.svm")
# a soft margin support vector machine
getParamSet(lrn.SVM)
rdesc<- makeResampleDesc("CV", iters=3)
# radial same as Gaussian when you tune; depending on what kernel you've seen
# Kernel: setting to radial(one type of kernel)
# cost is the c

# Parameter range
par.set<- makeParamSet(
  makeDiscreteParam("cost", values=c(2^-15, 2^-7, 1, 2^7, 2^15)),
  makeNumericParam("cost", lower=-5, upper=5, trafo=function(x) 2^x),  #uniform on the exponential scale
  makeNumericParam("gamma", lower=-15, )
)

# makeDiscreteParam("kernel", values=c("polynomial", "radial"))

makeLearner()

#parameter grid
ctrl<- makeTuneControlGrid(resolution=5)

# tuned SVM
makeTuneWrapper(lrn.SVM, cv3, f1, par.set, ctrl)  #f1:f1 score; can put mmce as well


# including a tuned learner in a benchmarking task
makeResampleDesc("CV", iters=5)
benchmark()