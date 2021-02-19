#The intent of the model is to use the independent variables (i.e., vehicle characteristics) to
#predict the dependent variable of MSRP.
#Out of 28 independent variables we only used 21 because we removed columns which we
#found uninformative due to the fact that they had the same value for all vehicles . To
#accommodate the "Colour" variable in the neural network model, we converted it to numeric
#by introducing dummy variables for each level of Colour.

#library(readxl)
#Car_Data <- read_excel("C:/Users/shriy/Desktop/Studies/2_Sem/DM_1/HW5/Car_Data.xlsx")
#View(Car_Data)
num_cols1 <- unlist(lapply(Car_Data, is.numeric))
num_cols1
unique(Car_Data$Colour)
CarNumeric <- Car_Data[, num_cols1]
str(CarNumeric)

Color_1 <- as.numeric(ifelse(Car_Data$Colour == "silver", 1, 0))
Color_2 <- as.numeric(ifelse(Car_Data$Colour == "grey", 1, 0))
Color_3 <- as.numeric(ifelse(Car_Data$Colour == "black", 1, 0))
Color_4 <- as.numeric(ifelse(Car_Data$Colour == "blue", 1, 0))
Color_5 <- as.numeric(ifelse(Car_Data$Colour == "red", 1, 0))
Color_6 <- as.numeric(ifelse(Car_Data$Colour == "green", 1, 0))

Color <- data.frame(Color_1, Color_2, Color_3, Color_4, Color_5, Color_6)


# Create vector of column Max and Min values
maxs_car <- apply(CarNumeric, 2, max) 
mins_car <- apply(CarNumeric, 2, min)

# Use scale() and convert the resulting matrix to a data frame
scaled.data <- as.data.frame(scale(CarNumeric, center = mins_car, scale = maxs_car - mins_car))
head(scaled.data)
summary(scaled.data)

data <- data.frame(scaled.data, Color)
str(data)
data <- data[, -c(8,9)]
data <- data[, -c(12,13)]
data <- data[, -18]

summary(data)

set.seed(123)
indx <- sample(2, nrow(data), replace = T, prob = c(0.8, 0.2))
train <- data[indx == 1, ]
test <- data[indx == 2, ]

# 3
# Single layer Neural Network
library(nnet)
nn  <- nnet(Price ~ ., data = train, linout=T, size=10, decay=0.01, maxit=1000)

summary(nn) 
library(devtools)
source_url('https://gist.githubusercontent.com/fawda123/7471137/raw/466c1474d0a505ff044412703516c34f1a4684a5/nnet_plot_update.r')
plot.nnet(nn)
# The darker lines are associated to the higher weights and gray lines are for small weights
nn.preds <- predict(nn, test)
nn.preds
MSE <- mean((nn.preds - test$Price)^2)
MSE

# Multi-Layer Neural Networks
library(RSNNS)
input <- train[,2:27]
mod<-mlp(input, train$Price, size=c(9,10,11),linOut=T,decay=0.01, maxit=1000)
plot.nnet(mod)
mod.preds <- predict(mod, test[,2:27])
mod.preds
MSE1 <- mean((mod.preds - test$Price)^2)
MSE1

# This model is generated using the "neuralnet" library and unlike RSNNS it displays the
# bias layer. There are 2 hidden layers having 8 and 5 neurons .

library(neuralnet)
mod1 <- neuralnet(train$Price ~ ., data = train, hidden = c(8,5), act.fct = "logistic",linear.output = TRUE)
plot(mod1)
mod1.preds <- predict(mod1, test)
mod1.preds
MSE2 <- mean((mod1.preds - test$Price)^2)
MSE2

# The MSE of a Single Layer Neural Network model is 0.174 and Multi-Layer Neural
# Network models are more complicated but the MSE is much lower. The MSE of the
# multi-layer perceptron having 3 hidden layers (MSE=0.014) is less than the MSE for
# multi-layer perceptron model (MSE=0.040) with two hidden layers.

# We ran Cross-validation to test different numbers of hidden neurons. The optimum size of
# neurons is 9 and the decay parameter is 0.01 for our case.
# Cross Validation
library(caret)
trainIndex <- createDataPartition(data$Price, p=.7, list=F)
data.train <- data[trainIndex, ]
data.test <- data[-trainIndex, ]
?expand.grid
my.grid <- expand.grid(.decay = c(0.5, 0.1), .size = c(9,10,11))
train.fit <- train(Price ~ ., data = data.train,
                      method = "nnet", maxit = 1000, tuneGrid = my.grid, trace = F, linout = 1)
train.fit


#Linear Regression
lmModel <- lm(train$Price ~ ., data = train)
summary(lmModel)

layout(matrix(c(1,2,3,4),2,2))
plot(lmModel)

lmModel.pred <- predict(lmModel, test)
lmModel.pred
MSE3 <- mean((test$Price - lmModel.pred)^2)
MSE3

# Based on the MSE performance of the neural network models vs. the linear
# regression model, we recommend the use of a neural network model to determine
# MSRP. The RSNSS model with three hidden layers returned the lowest MSE. 
# We recommend that the neural network model use the parameters of size = 9 and decay = 0.1.
