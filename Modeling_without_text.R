library(dplyr)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(gbm)
library(caTools)
library(ggplot2)
library(VIF)
library(car)
library(caret)

# load dataset 
data <- read.csv("data/final-airbnb.csv", sep = '\t')

# dataset with non-text column 
data1 <- select(data, Host.Response.Time, Host.Location, Calculated.host.listings.count,
                Host.Verifications, Property.Type, Room.Type, Accommodates, Bathrooms, Bedrooms, Beds, Bed.Type, Price,
                Guests.Included, Minimum.Nights, Maximum.Nights, Availability.365, Calendar.Updated,
                Number.of.Reviews, Review.Scores.Rating,
                Review.Scores.Accuracy, Review.Scores.Cleanliness,
                Review.Scores.Checkin, Review.Scores.Communication, Review.Scores.Location,
                Review.Scores.Value, Cancellation.Policy, Reviews.per.Month, duration_rev,
                Time.since.last.review, Number.of.Amenities,
                Host.Duration, crime, Neighbourhood.Cleansed)

# Split the dataset into training data (70%) and test data (30%)
set.seed(1)
train.ids = sample(nrow(data1), 0.70*nrow(data1))
train = data1[train.ids,]
test = data1[-train.ids,]

# Calcuate OSR2
OSR2 <- function(predictions, train, test) {
  SSE <- sum((test - predictions)^2)
  SST <- sum((test - mean(train))^2)
  r2 <- 1 - SSE/SST
  return(r2)
}
OSR2

#Baseline Model
mean_pred = mean(train$Price)
pred_base = rep(mean_pred,687)
MAE = mean(abs(test$Price-pred_base))
MAE
RMSE = sqrt(mean((test$Price - pred_base)^2))
RMSE
OSR2 = OSR2(predictions = pred_base, train$Price, test$Price)
OSR2

# Linear model
train.reg <- lm(Price ~ ., data = train)
summary(train.reg)

# Stepwise regression using backward 
BNBStep = step(train.reg, direction = "backward")
summary(BNBStep)
PredictStep = predict(BNBStep, newdata = test)
OSR2 = OSR2(PredictStep, test$Price, train$Price)
OSR2
MAE = mean(abs(test$Price-PredictStep))
MAE
RMSE = sqrt(mean((test$Price - PredictStep)^2))
RMSE

# CART with CV 
set.seed(1000)
train.cart <- train(Price ~ .,
                    data = train,
                    method = "rpart",
                    tuneGrid = data.frame(cp=seq(0, 0.1, 0.005)),
                    trControl = trainControl(method="cv", number=5),
                    metric = "RMSE")
train.cart$results
best_cp_b = train.cart$bestTune
best.cart = train.cart$finalModel
test.ctr.mm = as.data.frame(model.matrix(Price ~ ., data=test))
pred.best.cart = predict(best.cart, newdata = test.ctr.mm)
cfmatrix = table(test$Price, pred.best.cart)
cfmatrix

OSR2 = OSR2(pred.best.cart, test$Price, train$Price)
OSR2
MAE = mean(abs(test$Price-pred.best.cart))
MAE
RMSE = sqrt(mean((test$Price - pred.best.cart)^2))
RMSE

# Random Forest with CV
set.seed(1000)
train.rf = train(Price ~ .,
                 data = train,
                 method = "rf",
                 tuneGrid = data.frame(mtry = 1:15),
                 trControl = trainControl(method = "cv", number = 5, verboseIter = TRUE))
train.rf
train.rf$results
mod.rf = train.rf$finalModel
test.ctr = as.data.frame(model.matrix(Price ~ ., data=test))
predict.rf = predict(mod.rf, newdata = test.ctr)

OSR2 = OSR2(predict.rf, test$Price, train$Price)
OSR2
MAE = mean(abs(test$Price-predict.rf))
MAE
RMSE = sqrt(mean((test$Price - predict.rf)^2))
RMSE

# Boosting without CV
set.seed(232)
mod.boost <- gbm(Price ~ .,
                 data = train,
                 distribution = "gaussian",
                 n.trees = 1000,
                 shrinkage = 0.001,
                 interaction.depth = 2)

# NOTE: we need to specify number of trees to get a prediction for boosting
pred.boost <- predict(mod.boost, newdata = test, n.trees=1000)
# after you have trained a model, you can get the predictions for all earlier iterations as well, for example:

pred.boost.earlier <- predict(mod.boost, newdata = test, n.trees=330)

summary(mod.boost)

print("Boosting OSR2:")
OSR2(pred.boost, test$Price, train$Price)

# Boosting with CV
tGrid = expand.grid(n.trees = (1:75)*500, interaction.depth = c(1,2,4,6,8,10),
                    shrinkage = 0.001, n.minobsinnode = 10)

set.seed(232)
train.boost <- train(Price ~ .,
                     data = train,
                     method = "gbm",
                     tuneGrid = tGrid,
                     trControl = trainControl(method="cv", number=5, verboseIter = TRUE),
                     metric = "RMSE",
                     distribution = "gaussian")
train.boost #final values= optimal values that R selected
best.boost <- train.boost$finalModel
test.ctr.mm = as.data.frame(model.matrix(Price ~ . + 0, data=test)) 
pred.best.boost <- predict(best.boost, newdata = test.ctr.mm, n.trees = 11500) # can use same model matrix

ggplot(train.boost$results, aes(x = n.trees, y = Rsquared, colour = as.factor(interaction.depth))) + geom_line() + 
  ylab("CV Rsquared") + theme_bw() + theme(axis.title=element_text(size=18), axis.text=element_text(size=18)) + 
  scale_color_discrete(name = "interaction.depth")

print("Boosting OSR2:")
OSR2 = OSR2(pred.best.boost, test$Price, train$Price)
OSR2
MAE = mean(abs(test$Price-pred.best.boost))
MAE
RMSE = sqrt(mean((test$Price - pred.best.boost)^2))
RMSE

# CI Bootstrap
root_mean_squared_error <- function(data, index) {
  responses <- data$response[index]
  predictions <- data$prediction[index]
  RMSE <- sqrt(mean((responses - predictions)^2))
  return(RMSE)
}

mean_absolute_error <- function(data, index) {
  responses <- data$response[index]
  predictions <- data$prediction[index]
  MAE <- mean(abs(responses - predictions))
  return(MAE)
}

OS_R_squared <- function(data, index) {
  responses <- data$response[index]
  predictions <- data$prediction[index]
  baseline <- data$baseline[index]
  SSE <- sum((responses - predictions)^2)
  SST <- sum((responses - baseline)^2)
  r2 <- 1 - SSE/SST
  return(r2)
}

all_metrics <- function(data, index) {
  rmse <- root_mean_squared_error(data, index)
  mae <- mean_absolute_error(data, index)
  OSR2 <- OS_R_squared(data, index)
  return(c(rmse, mae, OSR2))
}

##### Stepwise ###### 
sw_test_set = data.frame(response = test$Price, prediction = PredictStep, baseline = mean(train$Price))
set.seed(1)
s_boot <- boot(sw_test_set, all_metrics, R = 10000)
s_boot

# get confidence intervals
boot.ci(s_boot, index = 1, type = "basic")
boot.ci(s_boot, index = 2, type = "basic")
boot.ci(s_boot, index = 3, type = "basic")

##### CART ######
CART_test_set = data.frame(response = test$Price, prediction = pred.best.cart, baseline = mean(train$Price))
set.seed(7191)
CART_boot <- boot(CART_test_set, all_metrics, R = 10000)
CART_boot

# get confidence intervals
boot.ci(CART_boot, index = 1, type = "basic")
boot.ci(CART_boot, index = 2, type = "basic")
boot.ci(CART_boot, index = 3, type = "basic")


##### Random Forests  ###### 
RF_test_set = data.frame(response = test$Price, prediction = pred.best.cart, baseline = mean(train$Price))
set.seed(892)
RF_boot <- boot(RF_test_set, all_metrics, R = 10000)
RF_boot

# get confidence intervals
boot.ci(RF_boot, index = 1, type = "basic")
boot.ci(RF_boot, index = 2, type = "basic")
boot.ci(RF_boot, index = 3, type = "basic")

##### Boosting  ###### 
BS_test_set = data.frame(response = test$Price, prediction = pred.best.boost, baseline = mean(train$Price))
all_metrics(BS_test_set, 1:687)
set.seed(7191)
Boost_boot <- boot(BS_test_set, all_metrics, R = 10000)
Boost_boot

# get confidence intervals
boot.ci(Boost_boot, index = 1, type = "basic")
boot.ci(Boost_boot, index = 2, type = "basic")
boot.ci(Boost_boot, index = 3, type = "basic")

