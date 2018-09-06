library(class)

### Question 2

# Read CSV from working directory into R

MyData <- read.csv(file="redwine.csv", header=TRUE, sep=",")

# Create binary variable final_quality using mean

MyData$final_quality <- with(ifelse(quality>mean(quality), 1, 0), data=MyData)

# Creating dataset without original quality attribute

myvars <- names(MyData) %in% c("quality")
fulldataset <- MyData[!myvars]

## a) Logistic Regression with all the data

glm.fit <- glm(final_quality ~ fixed.acidity+volatile.acidity+citric.acid+residual.sugar+chlorides+free.sulfur.dioxide+total.sulfur.dioxide+density+pH+sulphates+alcohol, data=fulldataset, family=binomial)
summary(glm.fit)


## b) Logistic Regression with the validation set approach

# i) Splitting data into train and val using 80/20 split

set.seed(1)
rows <- sample(x=nrow(fulldataset), size=.80*nrow(fulldataset))
trainset <- fulldataset[rows, ]
valset <- fulldataset[-rows, ]

# ii) Logistic Regression fit with only the training set

glm2.fit <- glm(final_quality ~ fixed.acidity+volatile.acidity+citric.acid+residual.sugar+chlorides+free.sulfur.dioxide+total.sulfur.dioxide+density+pH+sulphates+alcohol, data=trainset, family=binomial)
summary(glm2.fit)

# iii) Predictions for validation set: high_quality = 1, low_quality = 0

glm2.probs <- predict(glm2.fit, valset, type="response")
glm2.preds <- ifelse(glm2.probs>0.5, 1, 0)
confusion_matrix_glm2 <- table(valset$final_quality,glm2.preds)
print(confusion_matrix_glm2)

# iv) Fraction of misclassified observation in validation set

Misclassified_Predictions_fraction = (confusion_matrix_glm2[1,2]+confusion_matrix_glm2[2,1])/sum(confusion_matrix_glm2)
sprintf("Overall Fraction of Misclassified Predictions are: %f",Misclassified_Predictions_fraction)


### Question 3

## a. Creating function

set.seed(1)
boot.fn = function(data, index) return(coef(glm(final_quality ~ fixed.acidity+volatile.acidity+citric.acid+residual.sugar+chlorides+free.sulfur.dioxide+total.sulfur.dioxide+density+pH+sulphates+alcohol,data = fulldataset, family = binomial, subset = index)))

## b. Estimate standard errors

library(boot)
boot(fulldataset, boot.fn, 5000)

## c. Comparisons of the two sets of standard errors

# Both sets of the estimated standard errors obtained through glm() function and
# bootstrap function are very similar. For example, the std. error for
# free.sulfur.dioxide was 0.008236 with glm and with the bootsrap method, it was
# 0.008264231 (t7*). The std. error for residual.sugar was 0.053770 with glm,
# and 0.062763447 with boostrap (t5*).
