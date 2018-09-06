library(class)

# Read CSV from working directory into R

MyData <- read.csv(file="redwine.csv", header=TRUE, sep=",")

### Question 2

## a. 

# Numerical summary of data

str(MyData)
library(psych)
describe(MyData)
summary(MyData)

# Graphical summary of data

pairs(MyData, main="Scatter Plot of Variables")

# Correlation of attributes

z <- cor(MyData) 
z[lower.tri(z,diag=TRUE)]=NA  
z=as.data.frame(as.table(z)) 
z=na.omit(z)  
z=z[order(-abs(z$Freq)),]   
head(z, n=10)

  # After exploring the dataset, I generated a correlation matrix to understand
  # how the attributes are related to each other. I created a list of these
  # relationships, sorted them, and printed out the ones with the largest absolute
  # values. Obviously, a correlation of 1 would be the highest signify a perfect,
  # positive correlation whereas a -1 would signify a perfect, negative
  # correlation.

  # The highest correlation patterns in the data seem to be between fixed.acidity
  # & pH, fixed.acidity % citric.acid, fixed.acidity & density, and
  # freesulfur.dioxide & total.sulfur.dioxide. All of them have roughly a 0.68
  # correlation - only fixed.acidity and pH have an inverse relationship. Although
  # these correlations are not extremely high, they indicate that there is some
  # dependency of variables on each other which is potentially harmful to us when
  # building a model.

## b. 

# Create binary variable final_quality using mean

MyData$final_quality <- with(ifelse(quality>mean(quality), 1, 0), data=MyData)


# Creating dataset without original quality attribute

myvars <- names(MyData) %in% c("quality")
fulldataset <- MyData[!myvars]
head(fulldataset, n=1)

# Splitting data into train and test

set.seed(1)
rows <- sample(x=nrow(fulldataset), size=.80*nrow(fulldataset))
trainset <- fulldataset[rows, ]
testset <- fulldataset[-rows, ]

# Logistic Regression

glm.fit <- glm(final_quality ~ fixed.acidity+volatile.acidity+citric.acid+residual.sugar+chlorides+free.sulfur.dioxide+total.sulfur.dioxide+density+pH+sulphates+alcohol, data=trainset, family=binomial)
summary(glm.fit)

  # Atleast 4 of the predictors appear to be statistically significant. alcohol,
  # sulphates, total.sulfur.dioxide, and volative.acidity all have very small p
  # values meaning they have the largest impact on the response variable of final
  # quality.


## c. 

# Confusion Matrix

glm.probs <- predict(glm.fit, testset, type="response")
glm.preds <- ifelse(glm.probs>0.5, 1, 0)
confusion_matrix_glm <- table(testset$final_quality,glm.preds)
print(confusion_matrix_glm)

# Fraction of Correct Predictions

Correct_Predictions_fraction = (confusion_matrix_glm[1,1]+confusion_matrix_glm[2,2])/sum(confusion_matrix_glm)
sprintf("Overall Fraction of Correct Predictions are: %f",Correct_Predictions_fraction)

  # The confusion tells us about the performance of the model. There were True
  # Negatives (105) and True Positives (143) and, as can be seen above, constitute
  # about 77.5% of results. The rest were misclassified - so about 22.5%. 27
  # points were False Positives and 45 were False Negatives. This tells us our
  # logistic regression model is wrong about one fourth of the time and tends to
  # be too conservative. It is wrongly classifying good wine (in class 1 that have
  # quality above the mean) as bad wine (in class 0 with quality below mean) more
  # than it is classifying bad wine as good wine (although that is happening a
  # fair bit as well).

## d. 

variables <- which(names(fulldataset)%in%c("fixed.acidity","volatile.acidity","citric.acid","residual.sugar","chlorides","free.sulfur.dioxide","total.sulfur.dioxide","density","pH","sulphates","alcohol"))

test_error <- data.frame("k"=1:11)

set.seed(1)
for(k in 1:11)
  {
    knn.pred <- knn(train=trainset[, variables], test=testset[, variables], cl=trainset$final_quality, k=k)
    test_error$error[k]= round(sum(knn.pred!=testset$final_quality)/nrow(testset)*100,2)
  }

print(test_error)

# As seen above, the model with the lowest test error is when k=1 with an error
# of approximately 25 and therefore can be concluded to be performing the best
# on this dataset.


### Question 3

## a. 

# Split data into training and test - 80/20 

set.seed(1)

rows <- sample(x=nrow(fulldataset), size=0.8*nrow(fulldataset))
trainset <- fulldataset[rows, ]
testset <- fulldataset[-rows, ]

## b. 

# LDA

library (MASS)

lda.fit <- lda(final_quality ~ fixed.acidity+volatile.acidity+citric.acid+residual.sugar+chlorides+free.sulfur.dioxide+total.sulfur.dioxide+density+pH+sulphates+alcohol, data=trainset)
lda.pred <- predict(lda.fit, testset)
confusion_matrix_lda <- table(testset$final_quality, lda.pred$class)
print(confusion_matrix_lda)


test_error_lda <- sum(lda.pred$class!=testset$final_quality)/nrow(testset)
sprintf("The test error for LDA is: %f", test_error_lda)

## c. 

# QDA

qda.fit <- qda(final_quality ~ fixed.acidity+volatile.acidity+citric.acid+residual.sugar+chlorides+free.sulfur.dioxide+total.sulfur.dioxide+density+pH+sulphates+alcohol, data=trainset)
qda.pred <- predict(qda.fit, testset)
confusion_matrix_qda <- table(testset$final_quality, qda.pred$class)
print(confusion_matrix_qda)

test_error_qda <- sum(qda.pred$class!=testset$final_quality)/nrow(testset)
sprintf("The test error for QDA is: %f", test_error_qda)
