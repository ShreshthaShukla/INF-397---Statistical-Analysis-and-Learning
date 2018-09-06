
### Question 2

# a) 

set.seed(1)

# Generate predictor(X), noise(eps)

X = rnorm(100)
eps = rnorm(100)

# Generate coefficients(beta), response(Y)

beta0 = 3
beta1 = 2
beta2 = -3
beta3 = 0.3

Y = beta0 + beta1 * X + beta2 * X^2 + beta3 * X^3 + eps

## b)

# Build forward and backward selection models

library(leaps)

data <- data.frame(y = Y, x = X)

fwd_selection.model <- regsubsets(y ~ x + I(x^2) + I(x^3) + I(x^4) + I(x^5) + I(x^6), data = data, nvmax = 10, method = "forward")
bwd_selection.model <- regsubsets(y ~ x + I(x^2) + I(x^3) + I(x^4) + I(x^5) + I(x^6), data = data, nvmax = 10, method = "backward")

# Analysis of results 

fwd_selection.summary = summary(fwd_selection.model)
bwd_selection.summary = summary(bwd_selection.model)

par(mfrow = c(3, 2))

plot(fwd_selection.summary$cp, xlab = "Subset Size", ylab = "Forward Cp", pch = 20, type = "l")
points(which.min(fwd_selection.summary$cp), fwd_selection.summary$cp[which.min(fwd_selection.summary$cp)], pch = 4, col = "red", lwd = 3)

plot(bwd_selection.summary$cp, xlab = "Subset Size", ylab = "Backward Cp", pch = 20, type = "l")
points(which.min(bwd_selection.summary$cp), bwd_selection.summary$cp[which.min(bwd_selection.summary$cp)], pch = 4, col = "red", lwd = 3)

plot(fwd_selection.summary$bic, xlab = "Subset Size", ylab = "Forward BIC", pch = 20, type = "l")
points(which.min(fwd_selection.summary$bic), fwd_selection.summary$bic[which.min(fwd_selection.summary$bic)], pch = 4, col = "red", lwd = 3)

plot(bwd_selection.summary$bic, xlab = "Subset Size", ylab = "Backward BIC", pch = 20, type = "l")
points(which.min(bwd_selection.summary$bic), bwd_selection.summary$bic[which.min(bwd_selection.summary$bic)], pch = 4, col = "red", lwd = 3)

plot(fwd_selection.summary$adjr2, xlab = "Subset Size", ylab = "Forward Adjusted R2", pch = 20, type = "l")
points(which.max(fwd_selection.summary$adjr2), fwd_selection.summary$adjr2[which.max(fwd_selection.summary$adjr2)], pch = 4, col = "red", lwd = 3)

plot(bwd_selection.summary$adjr2, xlab = "Subset Size", ylab = "Backward Adjusted R2", pch = 20, type = "l")
points(which.max(bwd_selection.summary$adjr2), bwd_selection.summary$adjr2[which.max(bwd_selection.summary$adjr2)], pch = 4, col = "red", lwd = 3)

mtext("Plots of Cp, BIC, Adjusted R2 for forward and backward stepwise selection", side = 3, line = -2, outer = TRUE)

# Determine coefficients of models

fwd_selection.coef <- coefficients(fwd_selection.model, id=3)
bwd_selection.coef <- coefficients(bwd_selection.model, id=3)

print(fwd_selection.coef)
print(bwd_selection.coef)

# Comment on Results

# For both forward and backward stepwise selection, all three estimates of test
# error (Cp, BIC, Adjusted R2) selected a 3 predictor model. Both approaches
# also selected the same predictors : X, X^2, & X^5.

# c) 

library(glmnet)

# Built Lasso model

xmatrix <- model.matrix(y ~ x + I(x^2) + I(x^3) + I(x^4) + I(x^5) + I(x^6), data = data)[, -1]
lasso <- cv.glmnet(xmatrix, Y, alpha = 1)

par(mfrow = c(1, 1))
plot(lasso)

# Determine Optimal value of Lambda

bestlambda <- lasso$lambda.min
print(bestlambda)


# Fit Lasso model and get coefficients

lasso.fit <- glmnet(xmatrix, Y, alpha = 1)
predict(lasso.fit, s = bestlambda, type = "coefficients")[1:7, ]

# Comment on Results

# Lasso selected a 3 predictor model as well - using the same predictors : X, 
# X^2, X^5. The rest of the coefficients are zero. Although the values for the
# coefficients obtained from the lasso are similar in magnitude to the values
# selected through the forward and backward stepwise selection, they are not
# exactly the same.

