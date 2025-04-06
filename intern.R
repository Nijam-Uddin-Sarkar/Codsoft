#####Task-1
# Load necessary libraries
library(tidyverse)
library(caret)
library(rpart)
library(rpart.plot)

# Load the Titanic dataset
titanic <- read_csv("~/Internship/Titanic-Dataset.csv")

# View dataset structure
str(titanic)

# Select relevant columns and preprocess data
titanic <- titanic %>%
  select(Survived, Pclass, Sex, Age, SibSp, Parch, Fare, Embarked) %>%
  mutate(
    Survived = as.factor(Survived),  # Convert target variable to factor
    Pclass = as.factor(Pclass),
    Sex = as.factor(Sex),
    Embarked = as.factor(Embarked)
  )

# Handle missing values by replacing NAs with median/mode
titanic$Age[is.na(titanic$Age)] <- median(titanic$Age, na.rm = TRUE)
titanic$Embarked[is.na(titanic$Embarked)] <- "S"

# Split data into training (80%) and testing (20%) sets
set.seed(123)
trainIndex <- createDataPartition(titanic$Survived, p = 0.8, list = FALSE)
trainData <- titanic[trainIndex, ]
testData <- titanic[-trainIndex, ]

# ---- Logistic Regression Model ----
glm_model <- glm(Survived ~ ., data = trainData, family = binomial)

# Predict on test set
glm_pred <- predict(glm_model, testData, type = "response")

# Convert probabilities to class labels (0 or 1)
glm_pred_class <- ifelse(glm_pred > 0.5, "1", "0")
glm_pred_class <- as.factor(glm_pred_class)

# Evaluate GLM model
glm_conf_mat <- confusionMatrix(glm_pred_class, testData$Survived)
print("Logistic Regression Confusion Matrix:")
print(glm_conf_mat)

# ---- Classification Tree Model ----
tree_model <- rpart(Survived ~ ., data = trainData, method = "class")

# Visualize the decision tree
rpart.plot(tree_model, type = 3, extra = 101, fallen.leaves = TRUE)

# Predict on test set
tree_pred <- predict(tree_model, testData, type = "class")

# Evaluate Decision Tree model
tree_conf_mat <- confusionMatrix(tree_pred, testData$Survived)
print("Classification Tree Confusion Matrix:")
print(tree_conf_mat)

#####Task-3

# Load necessary libraries
library(tidyverse)
library(caret)
library(nnet)  # For multinomial logistic regression
library(rpart)
library(rpart.plot)

# Load the built-in Iris dataset
data(iris)

# View dataset structure
str(iris)

# Rename target variable for convenience
iris$Species <- as.factor(iris$Species)

# Split data into training (80%) and testing (20%) sets
set.seed(123)  # For reproducibility
trainIndex <- createDataPartition(iris$Species, p = 0.8, list = FALSE)
trainData <- iris[trainIndex, ]
testData <- iris[-trainIndex, ]

# ---- Logistic Regression (Multinomial) ----
glm_model <- multinom(Species ~ ., data = trainData)

# Predict on test set
glm_pred <- predict(glm_model, testData)

# Evaluate Logistic Regression model
glm_conf_mat <- confusionMatrix(glm_pred, testData$Species)
print("Multinomial Logistic Regression Confusion Matrix:")
print(glm_conf_mat)

# ---- Decision Tree Model ----
tree_model <- rpart(Species ~ ., data = trainData, method = "class")

# Visualize the Decision Tree
rpart.plot(tree_model, type = 3, extra = 101, fallen.leaves = TRUE)

# Predict on test set
tree_pred <- predict(tree_model, testData, type = "class")

# Evaluate Decision Tree model
tree_conf_mat <- confusionMatrix(tree_pred, testData$Species)
print("Decision Tree Confusion Matrix:")
print(tree_conf_mat)

#####Task-5
# Load necessary libraries
library(caret)          # For data splitting and model evaluation
library(ggplot2)        # For visualization (if needed)
library(ranger)         # For memory-efficient Random Forest (ranger)
library(ROCR)           # For evaluation metrics

# Load the dataset
data <- read.csv("Internship/creditcard.csv")

# Convert 'Class' to a factor (if it is not already)
data$Class <- as.factor(data$Class)

# Explore the data
str(data)
summary(data)

# Data preprocessing: Normalize and handle missing values
# Normalize numeric columns (excluding the 'Class' and 'Time' columns)
numeric_cols <- sapply(data, is.numeric)
data[numeric_cols] <- scale(data[numeric_cols])

# Split the data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(data$Class, p = 0.8, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

# Optional: Subsample the training data (50%) to reduce memory usage
set.seed(123)
sampled_data <- trainData[sample(nrow(trainData), size = nrow(trainData) * 0.5), ]

# Train the Random Forest model using the ranger package (more memory-efficient)
rf_model <- ranger(Class ~ ., data = sampled_data, num.trees = 100, importance = 'impurity')

# Summarize the model
print(rf_model)

# Predicting on the test set using the ranger model
rf_predictions <- predict(rf_model, data = testData)$predictions

# Ensure the levels of both predicted and actual values are the same (0 and 1)
rf_predictions <- factor(rf_predictions, levels = c(0, 1))
testData$Class <- factor(testData$Class, levels = c(0, 1))

# Manual computation of confusion matrix elements for Random Forest
TP_rf <- sum(testData$Class == 1 & rf_predictions == 1)
TN_rf <- sum(testData$Class == 0 & rf_predictions == 0)
FP_rf <- sum(testData$Class == 0 & rf_predictions == 1)
FN_rf <- sum(testData$Class == 1 & rf_predictions == 0)

# Compute metrics: Accuracy, Precision, Recall, F1-Score
accuracy_rf <- (TP_rf + TN_rf) / length(testData$Class)
precision_rf <- TP_rf / (TP_rf + FP_rf)
recall_rf <- TP_rf / (TP_rf + FN_rf)
f1_rf <- 2 * (precision_rf * recall_rf) / (precision_rf + recall_rf)

cat("\nRandom Forest (ranger) Model - Metrics:\n")
cat("Accuracy:", accuracy_rf, "\n")
cat("Precision:", precision_rf, "\n")
cat("Recall:", recall_rf, "\n")
cat("F1 Score:", f1_rf, "\n")

