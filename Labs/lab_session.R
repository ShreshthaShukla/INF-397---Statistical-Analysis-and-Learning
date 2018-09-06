x <- c(1,2,3,4)
length(x)
mode(x)
y <-("apple")
length(y)
z <- c("a", "b")
length(z)
x <- list(name = "Jack", score = 99)
x$name
n <- cbind (c(1,2),c(3,4))
n
?rbind
x <- matrix (seq(1,12), nrow=3, ncol=4)
x
x <- matrix (seq(1,12), nrow=3, ncol=4, byrow=TRUE)
x
x[1,2]
x[1,]
x[1:3,2:4]
x[-4,-1]
dim(x)
d <- data.frame(list(name=c("Jack", "Rose"), score = c(99,98)))
d
d$score
View(d)
my_vector <- c(1:10)
my_matrix <- matrix(seq(1:9), ncol=9)
dd<-d[1,]
l <- list(my_vector,my_matrix, dd)
l
x <- runif(50)
x
y <- rnorm(50, mean=50, sd=0.1)
y
set.seed(12345)
y <- rnorm(50, mean=50, sd=0.1)
y
z <- rnorm(50)
z
mean(z)
sd(z)
getwd()
setwd("C:/Users/sanch/OneDrive/Documents/Graduate School/UT-Austin - MSIS/INF 397 - Statistical Analysis and Learning/Data")
mydata = read.csv("austin_house_price.csv")
mydata
x<-rnorm(100)
y<-rnorm(100)
hist(x)
plot(x,y, xlab="sdfs", ylab="dsfsdf")
