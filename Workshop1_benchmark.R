data(iris)
head(iris)
summary(iris)

# Hypothesis in species association questions
library(lattice)
densityplot(~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, data=iris, width = 1)
pairwise.wilcox.test(iris$Sepal.Length, iris$Species)

library(mlr)
# classifier leans a function from a four dimensional space

help("subsetTask")
subsetTask(iris.task, features = c("Sepal.Length","Sepal.Width"))

help("makeLearner")
set.seed(4242)  # important!

help("plotLearnerPrediction")  # just to show how answer the defer
# retraining the data - neural network; you get different answers
# highlight one issue: variation from the data and the fitting process and ...
# extrapolation

help("makeClassifTask")
help("makeResampleDesc")

#list(...,mmce.se,...)
help(benchmark)
# benchmark table doesn't tell us the significance
# add a measurement to produce the confidence interval: mmce.se

learners<- list(makeLearner("regr.lm", id='lm'),
                makeLearner("regr.randomForest", id="RF"))
task<- makeRegrTask(id="mtcars", data=mtcars, target = "mpg")
measures <- list(rmse, mae, timetrain)
valsetup<- makeResampleDesc("CV", iters=10)

bmresults<-benchmark(learners, task, valsetup, measures)
# output a table
getBMRAggrPerformances(bmresults) # aggregate performance table
getBMRPerformances(bmresults) # performance per re-sample split
getBMRPredictions(bmresults) # out-of-sample predictions


