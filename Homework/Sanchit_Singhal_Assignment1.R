
### Problem 2 - MLR on dataset

# Read CSV from working directory into R

MyData <- read.csv(file="austin_house_price.csv", header=TRUE, sep=",")

# a. Scatterplot matrix with all variables in dataset

pairs(MyData)

# b. Matrix of correlations of all variables

cor(MyData)

# c. Multiple Linear Regression

lm.fit=lm(SalePrice~., data=MyData)
summary(lm.fit)

par(mfrow = c(2, 2))
plot(lm.fit)

  # Relationship between predictors and response: 

  # By testing the null hypothesis of that there is no relationship, we can
  # reject it by looking at the p-value corresponding to the F-statistic. In
  # this case, it is very small (<2.2e-16) which means there appears to be a
  # strong relationship between "SalePrice" and atleast some of the predictors.
  # Indeed, by looking at the regression coefficients it can be seen that
  # "GarageCars", "BsmtFullBath", "TotRmsAbvGrd", "OverallQual" all have small
  # p-values and are therefore statistically significant.

  # Coefficient for the age variable:
  
  # The regression coefficient for the age, -248.83, suggests that for every 1
  # unit in age (presumably a year), SalePrice decreases by the coefficient. In
  # other words, the price falls every year which makes sense because property is
  # usuallly more expensive the newer it is.


# d. Transformation of the variables

par(mfrow = c(2, 2))
plot(log(MyData$GarageCars), MyData$SalePrice)
plot(sqrt(MyData$BsmtFullBath), MyData$SalePrice)
plot((MyData$TotRmsAbvGrd)^2, MyData$SalePrice)
plot((MyData$OverallQual)^2, MyData$SalePrice)

  # Comment on findings:

  # I decided to transform variables that had the highest statistically
  # significance (lowest p-values) because they have the greastest impact on the
  # SalesPrice. After trying out some transformation, I believe the square of the
  # overall quality gives the most linear looking plot.


### Problem 3 - SLR on simulated data

set.seed(1)
par(mfrow = c(1, 1))

# a. Generation of Feature X

x = rnorm(100)

# b. Generation of Feature eps

eps = rnorm(100, 0, sqrt(0.25))

# c. Generation of response

y = y = -1 + 0.5*x + eps
length(y)


  # Length of vector, Y:
  
  # The length of vector, Y, is 100 which makes sense since it a linear function
  # of 2 sets of 100 values

  # Values for B0 & B1:

  # B0 = -1, B1 = 0.5 as seen from the original equation


# d. Scatterplot

plot(x, y)

  # Comment on observations:
  
  # The relationship between x & y has a positive, linear slope with some
  # variance due to the noise introduced by the eps variable.

# e. Least Square Linear Model

lm.fit2 <- lm(y ~ x)
summary(lm.fit2)

  # Comment on Model:
  
  # The model has a large F-statistic with a small p-value (4.583e-15) and so the null
  # hypothesis can be rejected. This makes sense to me as we know y was indeed
  # generated using x and therefore, the two definitively have a relationship.

  # How do B^0 and B^1 compare to B0 and B1:
  
  # The constructed values for B^0 (-1.019) and B^1 (0.499) were very close to
  # the true values of -1 and 0.5. This means the linear regression model does a
  # great job modelling the relationship between x & y.

# f. Polynomial Regression Model

lm.fit2_sq = lm(y~x+I(x^2))
summary(lm.fit2_sq)

  # Does quadratic term improve the model fit:
  
  # There is evidence that the model fit has increased slightly as the RSE has
  # decreased and the R^2 is higher. However, when taking into account the large
  # p-value for the x^2 coefficient, it can be concluded that x^2 does not have
  # a relationship with y and the model is most likely overfitting the training
  # data by learning too much of the noise.

# g. Reduction of Noise

set.seed(1)
eps2 = rnorm(100, 0, 0.125)
x2 = rnorm(100)
y2 = -1 + 0.5*x2 + eps2
plot(x2, y2)
lm.fit3 = lm(y2~x2)
summary(lm.fit3)

  # Description of Results
  
  # By decreasing the variance of the normal distribution that generates the
  # error term, eps, we are able to reduce noise. The coefficients for B0 and B1
  # remain very similar which tells us that the model remained the same.
  # However, the RSE has significantly decreased, and R^2 has increased which
  # means the model fits extremely well. Again, this makes sense because the
  # underlying data is near-perfect with very little error.





