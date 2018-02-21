library(pROC)
library(gbm)
library(randomForest)
library(caret)
library(readr)
library(rpart.plot)
library(caTools);library(rpart)


#The dataset has been collected and analysed during a research collaboration of Worldline and the Machine Learning Group (http://mlg.ulb.ac.be) of ULB (Université Libre de Bruxelles) on big data mining and fraud detection. More details on current and past projects on related topics are available on http://mlg.ulb.ac.be/BruFence and http://mlg.ulb.ac.be/ARTML

creditcard <- read_csv("D:/Projects/Machine learning and data science + R/Credit Card Fraud/creditcardfraud/creditcard.csv")

apply(creditcard, 2, anyNA)  # checking if there is any NA
table(creditcard$Class)
length(row(creditcard))
length(col(creditcard))
Amelia::AmeliaView()


set.seed(4495)
creditcard$Time <- NULL
creditcard[is.na(creditcard)] = -9999

# Replace NA with mean
replaceNAWithMean <- function(data) {
  for(i in 1:ncol(data)){
    data[is.na(data[,i]), i] <- mean(data[,i], na.rm = TRUE)
  }
}
replaceNAWithMean(creditcard)


#creditcard1 <- rfImpute(creditcard$Class~.,creditcard)

## -------- Imputing missing values with mean values --------
require(plyr)
require(Hmisc)

creditcard <- ddply(creditcard, creditcard$V1, mutate, imputed.value = impute(value, mean))

# ------ creating partition ------------
set.seed(4495)
t<-createDataPartition(p=0.5,y=creditcard$Class,list = F)
training<-creditcard[t,]
testing<-creditcard[-t,]

table(training$Class)
table(testing$Class)
#training$Class <- as.factor(training$Class)
class(training$Class)

#XXXXXXXXXXXXXXXXXXXXXXXX----- Visualization ----XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
library(ggplot2)
Y <- creditcard$Class
Y <- as.factor(Y)
ggplot(creditcard,aes(x = Y)) + geom_bar(aes(fill = Y)) + xlab('Class')
training$Class <- as.factor(training$Class)
ggplot(training,aes(x = training$Class)) + geom_bar(aes(fill = training$Class)) + xlab('Class')
(table(Y)[1]/length(Y))*100
(table(Y)[2]/length(Y))*100

#XXXXXXXXXXXXXXXXXXXXX------ Generatng Synthetic data ------XXXXXXXXXXXXXXXXXXXXXXX
library(ROSE)
attach(training)
set.seed(4495)
training_Rose <- ROSE(Class~.,data=training,seed = 4495)$data
training_Rose$Class <- as.factor(training_Rose$Class)
ggplot(training_Rose,aes(x = Class)) + geom_bar(aes(fill = Class))

## ------------------- Undersampling ------------------------
training <- na.omit(training)
attach(training)
training$Class <- as.factor(training$Class)
training_under <- ovun.sample(Class~.,data = training,method = "under",
                              N=800,seed=4495)$data
ggplot(training_under,aes(x = Class)) + geom_bar(aes(fill = Class))

## -------------------- oversampling ------------------------

training_over <- ovun.sample(Class~.,data = training,method = "over",
                             N=202404,seed=4495)$data
ggplot(training_over,aes(x = Class)) + geom_bar(aes(fill = Class))+ggtitle("Over Sampling")

## ---------- SMOTE -------------------
training_smote <- SMOTE(training$Class~.,data = training,perc.over = 1000,perx.under = 5000)

##   using mlr

library(mlr)
training_smote <- smote(training$Class,rate = 5,nn=5)

#debug(SMOTE(training$Class~.,data = training,perc.over = 1000,perx.under = 5000))
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX----Random Forest----XXXXXXXXXXXXXXXXXXXXXXXXX

set.seed(4495)
rand_model <- randomForest(training$Class~., data = training, ntree = 200, 
                           importance = T,proximity = T)

varImpPlot(rand_model)

res <- predict(rand_model, testing)
accuracy<-(1-mean(res!=testing$Class))*100
accuracy 


#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX ----- GBM ---------XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

library(gbm)

# Get the time to train the GBM model
system.time(
  gbm.model <- gbm(training$Class ~ .
                   , distribution = "bernoulli"
                   , data = rbind(training, testing)
                   , n.trees = 500
                   , interaction.depth = 3
                   , n.minobsinnode = 100
                   , shrinkage = 0.01
                   , bag.fraction = 0.5
                   , train.fraction = nrow(training) / (nrow(training) + nrow(testing))
  )
)
# Determine best iteration based on test data
best.iter = gbm.perf(gbm.model, method = "test")

# Get feature importance
gbm.feature.imp = summary(gbm.model, n.trees = best.iter)

# Plot and calculate AUC on test data
gbm.test = predict(gbm.model, newdata = test, n.trees = best.iter)
auc.gbm = roc(test$Class, gbm.test, plot = TRUE, col = "red")
print(auc.gbm)

#XXXXXXXXXXXXXXXXXXXXXX-- Logistic Regresion ----XXXXXXXXXXXXXXXXXXXXXXXXXXXxx

attach(training)
attach(training_)
#training$Class <- Y
log <- glm(Class~., data = training,family=binomial)
log2 <- glm(training_Rose$Class~., data = training_Rose,family=binomial(logit))
log3 <- glm(training_under$Class~.,data = training_under,family=binomial(logit))
log4 <- glm(training_over$Class~.,data = training_over,family=binomial(logit))

ans <- predict(log4, testing,type="response")
ans <- round(ans)
accuracy <- (1-mean(ans != testing$Class))*100
accuracy  # 99.9 %
confusionMatrix(ans,testing$Class)
mat <- as.matrix(confusionMatrix(ans,testing$Class))


print(roc(testing$Class,ans))
plot(roc(testing$Class,ans),main = "Logistic regression ROC curve(UnderSampling)")
# Area under the curve: 0.8149

#XXXXXXXXXXXXXXXXXXXX--- Treebag -----XXXXXXXXXXXXXXX
control <- trainControl(method="cv",number=5)
tb_model <- train(training$Class~.,data = training,method = "treebag",
                  trControl=control)

tb_pred <- predict(testing,tb_model)
roc(tb_pred,testing$Class)

#XXXXXXXXXXXXXXXXXXX ----Decision tree model- -----XXXXXXXXXXXXXXXXX
library(rpart)
set.seed(4495)
tree.model <- rpart(Class ~ ., data = training, method = "class", minbucket = 20)
tree.model2 <- rpart(training_Rose$Class ~ ., data = training, method = "class", minbucket = 20)
tree.model3 <- rpart(training_under$Class ~ ., data = training_under, method = "class", minbucket = 20)
tree.model4 <- rpart(training_over$Class~.,data = training_over,method = "class",minbucket = 20)
prp(tree.model)
prp(tree.model4)
tree.predict <- predict(tree.model4,testing,type = "class")
(1-mean(tree.predict != testing$Class))*100
confusionMatrix(tree.predict,testing$Class)
mat <- as.matrix(confusionMatrix(tree.predict,testing$Class))
print(roc(testing$Class,ans))
plot(roc(testing$Class,ans),main = "Decision Tree ROC curve(UnderSampling)")

#XXXXXXXXXXXXXXXXXXXXXXX----- xgboost ------XXXXXXXXXXXXXXXXXXXXXXXX
library(xgboost)

set.seed(4495)


Y <- training$Class
Y1 <- training_Rose$Class
Y1 <- as.integer(Y1)-1
Y <- as.integer(Y)
Y2 <- as.integer(training_under$Class)-1
training$Class <- NULL
training_Rose$Class <- NULL
training_under$Class = NULL

Y3 <- as.integer(training_over$Class) - 1
training_over$Class <- NULL

clf.test <- microbenchmark(
  clf <-  xgboost(data       = data.matrix(training_under),
                       label       = Y2,
                       nrounds     = 30,
                       eta = 0.1,
                       max.depth = 7,
                       objective   = "binary:logistic",
                       nthread      = 3,
                       eval_metric = "auc")
                       , times = 3L)


xgb_importance <- xgb.importance(model = clf)
xgb_importance

ans2 <- predict(clf,data.matrix(testing))
ans2 <- round(ans2)
accuracy_xgboost <- (1-mean(ans2!=testing$Class))*100
accuracy_xgboost #99.95 %
confusionMatrix(ans2,testing$Class)
#          Reference
#Prediction      0      1
#         0 142130     56
#         1     16    201
auc.xgb.speed = roc(testing$Class, ans2, plot = TRUE, col = "blue")
print(auc.xgb.speed)
plot(auc.xgb.speed,main = "xgBoost ROC Curve (Undersampling)")
#Area under the curve: 0.897

#XXXXXXXXXXXXXXXXXXXXXXX----- LightGBM ------XXXXXXXXXXXXXXXXXXXXXXXX
library(devtools)
options(devtools.install.args = "--no-multiarch") # if you have 64-bit R only, you can skip this
install_github("Microsoft/LightGBM", subdir = "R-package")
devtools::install_github("Microsoft/LightGBM", ref = "1b7643b", subdir = "R-package")

library(lightgbm, quietly=TRUE)
params.lgb = list(
  objective = "binary"
  , metric = "auc"
  , min_data_in_leaf = 1
  , min_sum_hessian_in_leaf = 100
  , feature_fraction = 1
  , bagging_fraction = 1
  , bagging_freq = 0
)

lgb.bench = microbenchmark(
  lgb.model <- lgb.train(
    params = params.lgb
    , data = lgb.train
    , valids = list(test = lgb.test)
    , learning_rate = 0.1
    , num_leaves = 7
    , num_threads = 2
    , nrounds = 500
    , early_stopping_rounds = 40
    , eval_freq = 20
  )
  , times = 5L
)

#XXXXXXXXXXXXXXXXXXXXXX---- SVM ----XXXXXXXXXXXXXXXXXXXXXXXXXXXxx

library(e1071)

model_svm <- svm(Class~.,data = training,cost=1,gamma=1)
