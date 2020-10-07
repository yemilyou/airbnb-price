library(tm)
library(SnowballC)
library(MASS)
library(caTools)
library(dplyr)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(gbm)
library(ggplot2)
library(VIF)
library(car)

#load the data
airbnb <- read.csv("data/final-airbnb.csv", sep = '\t')

#converting listing names, summary, space, description to a "corpus"
summary_corpus = Corpus(VectorSource(airbnb$Summary))
house_rules_corpus = Corpus(VectorSource(airbnb$House.Rules))
host_about_corpus = Corpus(VectorSource(airbnb$Host.About))

#lower case
summary_corpus = tm_map(summary_corpus, tolower)
house_rules_corpus = tm_map(house_rules_corpus, tolower)
host_about_corpus = tm_map(host_about_corpus, tolower)

# remove punctuation
summary_corpus = tm_map(summary_corpus, removePunctuation)
house_rules_corpus = tm_map(house_rules_corpus, removePunctuation)
host_about_corpus = tm_map(host_about_corpus, removePunctuation)

# remove stop words
summary_corpus = tm_map(summary_corpus, removeWords, stopwords("english"))
house_rules_corpus = tm_map(house_rules_corpus, removeWords, stopwords("english"))
host_about_corpus = tm_map(host_about_corpus, removeWords, stopwords("english"))

#stem
summary_corpus = tm_map(summary_corpus, stemDocument)
house_rules_corpus = tm_map(house_rules_corpus, stemDocument)
host_about_corpus = tm_map(host_about_corpus, stemDocument)

#create a word count matrix
summary_frequencies = DocumentTermMatrix(summary_corpus)
house_rules_frequencies = DocumentTermMatrix(house_rules_corpus)
host_about_frequencies = DocumentTermMatrix(host_about_corpus)

#sparsity (.9)
summary_sparse = removeSparseTerms(summary_frequencies, 0.90)
house_rules_sparse = removeSparseTerms(house_rules_frequencies, 0.90)
host_about_sparse = removeSparseTerms(host_about_frequencies, 0.90)

#creating dataframe
summary_dat = as.data.frame(as.matrix(summary_sparse))
colnames(summary_dat) = make.names(colnames(summary_dat))

house_rules_dat = as.data.frame(as.matrix(house_rules_sparse))
colnames(house_rules_dat) = make.names(colnames(house_rules_dat))

host_about_dat = as.data.frame(as.matrix(host_about_sparse))
colnames(host_about_dat) = make.names(colnames(host_about_dat))

#putting everything into a dataframe
data_new <- data.frame(summary_dat, house_rules_dat, host_about_dat)
write.csv(data_new, "Desktop/text_data.csv")
data_new$Price <- airbnb$Price

# Split data into training and testing sets
set.seed(1)  # So we get the same results
spl = sample.split(data_new$Price, SplitRatio = 0.7)

textTrain = data_new %>% filter(spl == TRUE)
textTest = data_new %>% filter(spl == FALSE)

# funciton for calculating OSR2
OSR2 <- function(predictions, test, train) {
  SSE <- sum((test - predictions)^2)
  SST <- sum((test - mean(train))^2)
  r2 <- 1 - SSE/SST
  return(r2)
}

### MODEL BUILDING:


# 1) Basic Random Forests:
text_basicRF = randomForest(Price ~ ., data=textTrain)

Predict_simpleRF = predict(text_basicRF, newdata = textTest)

print("text Basic RF OSR2:")
OSR2(Predict_simpleRF, textTest$Price, textTrain$Price)

#metrics 
MAE_simpleRF = mean(abs(textTest$Price-Predict_simpleRF))
MAE_simpleRF
RMSE_simpeRF = sqrt(mean((textTest$Price - Predict_simpleRF)^2))
RMSE_simpeRF



# 2) Cross-validated CART model
set.seed(1)
text_train.cart = train(Price ~ .,
                   data = textTrain,
                   method = "rpart",
                   tuneGrid = data.frame(cp=seq(0, 0.1, 0.005)),
                   trControl = trainControl(method="cv", number=5),
                   metric = "RMSE")
text_train.cart
text_train.cart$results

best_text_train_cart=text_train.cart$bestTune
best_text.cart = text_train.cart$finalModel
prp(best_text.cart)
text.test.ctr.mm = as.data.frame(model.matrix(Price ~ ., data=textTest))
text.pred.best.cart = predict(best_text.cart, newdata = text.test.ctr.mm)

print("CV CART2 OSR2:")
OSR2(text.pred.best.cart, textTest$Price, textTrain$Price)


#metrics 
MAE_cvcart = mean(abs(textTest$Price-text.pred.best.cart))
MAE_cvcart
RMSE_cvcart = sqrt(mean((textTest$Price - text.pred.best.cart)^2))
RMSE_cvcart

text.cart.plot2 <- ggplot(text_train.cart$results, aes(x = cp, y = RMSE)) + geom_point(size = 2) + geom_line() + 
  ylab("CV RMSE") + theme_bw() + 
  theme(axis.title=element_text(size=18), axis.text=element_text(size=18))
ggsave("Desktop/text.cvcart.plot.2", text.cart.plot2)

# 3) Cross validated RF
set.seed(1)
text.train.rf = train(Price ~ .,
                  data = textTrain,
                  method = "rf",
                  tuneGrid = data.frame(mtry = 1:20),
                  trControl = trainControl(method = "cv", number = 5, verboseIter = TRUE), metric = "RMSE")
text.train.rf
text.train.rf$results

text.predict.rf = predict.train(text.train.rf, newdata = textTest)

ggplot(text.train.rf$results, aes(x = mtry, y = RMSE)) + geom_point(size = 2) + geom_line() + 
  ylab("CV RMSE") + theme_bw() + 
  theme(axis.title=element_text(size=18), axis.text=element_text(size=18))

print("CV RF for text OSR2:")
OSR2(text.predict.rf, textTest$Price, textTrain$Price)

#other metrics
MAE_cvRF = mean(abs(textTest$Price-text.predict.rf))
MAE_cvRF
RMSE_cvRF = sqrt(mean((textTest$Price - text.predict.rf)^2))
RMSE_cvRF

importance(text.train.rf$finalModel)
varImpPlot(text.train.rf$finalModel)


# 4) stepwise regression
text.train.reg <- lm(Price ~ ., data = textTrain)
summary(text.train.reg)
vif(text.train.reg)

text.step = step(text.train.reg, direction = "backward")

Predict.text.Step = predict(text.step, newdata = textTest)
print("backward stepwise for text OSR2:")
OSR2(Predict.text.Step, textTest$Price, textTrain$Price)

#other metrics
MAE_step = mean(abs(textTest$Price-Predict.text.Step))
MAE_step
RMSE_step = sqrt(mean((textTest$Price - Predict.text.Step)^2))
RMSE_step

#Boosting
# Boosting with CV
tGrid = expand.grid(n.trees = (1:75)*500, interaction.depth = c(1,2,4,6,8,10),
                    shrinkage = 0.001, n.minobsinnode = 10)

set.seed(1)
train.boost <- train(Price ~ .,
                     data = textTrain,
                     method = "gbm",
                     tuneGrid = tGrid,
                     trControl = trainControl(method="cv", number=5, verboseIter = TRUE),
                     metric = "RMSE",
                     distribution = "gaussian")
train.boost #final values= optimal values that R selected
best.boost <- train.boost$finalModel
test.ctr.mm = as.data.frame(model.matrix(Price ~ . + 0, data=textTest)) 
pred.best.boost <- predict(best.boost, newdata = textTest, n.trees = 11500) # can use same model matrix

ggplot(train.boost$results, aes(x = n.trees, y = Rsquared, colour = as.factor(interaction.depth))) + geom_line() + 
  ylab("CV Rsquared") + theme_bw() + theme(axis.title=element_text(size=18), axis.text=element_text(size=18)) + 
  scale_color_discrete(name = "interaction.depth")

print("Boosting OSR2:")
OSR2(pred.best.boost, textTest$Price, textTrain$Price)

#other metrics
MAE_boost = mean(abs(textTest$Price-pred.best.boost))
MAE_boost
RMSE_boost = sqrt(mean((textTest$Price - pred.best.boost)^2))
RMSE_boost