# Practical Machine Learning Course Project
Rail Suleymanov  
Sunday, June 21, 2015  

The goal of this project is to take measurement data from motion-tracking devices and use it to predict the manner in which their owners did the exercise ("classe" variable in data set). In this document I will describe how I built the model, cross-validation methods and report the out-of-sample error. Developed prediction model will be used to predict 20 different test cases.

Let's start with loading data. Here 'data' is one we know the "classe" variable for and it will be used for creating training/validation sets. 'testing' is 20 test cases data set (one we don't know to which class it belongs).


```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.1.3
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```
## Warning: package 'ggplot2' was built under R version 3.1.2
```

```r
data <- read.csv("../pml-training.csv")
testing <- read.csv("../pml-testing.csv")
dim(data)
```

```
## [1] 19622   160
```

Examining data/testing sets shows that there are a lot of missing data, and as far as we have a lot of data samples in training set (15699) and a lot of features (159), it would be worth removing some. This can be also useful since some features in testing set don't have any necessary information and consist only of NA's. We can't omit all NA's since it will result in quite a poor data set:


```r
poor_data <- na.omit(data)
dim(poor_data)
```

```
## [1] 406 160
```

Let's find all feature indexes that are NA in testing set and remove these features from both data/testing:


```r
na_inds <- c()
for (i in 1:dim(testing)[2]) {
    vec <- testing[, i]
    if (length(vec) == length(vec[is.na(vec)]))
        na_inds <- c(na_inds, i)
}
print(length(na_inds))
```

```
## [1] 100
```

```r
data <- data[, -na_inds]
testing <- testing[, -na_inds]
```

Now the data set is ready for partitioning. I'll divide it to 2 classes - training (80% of data) and validating (20%):


```r
i_train <- createDataPartition(y=data$classe, p=0.8, list=FALSE)
training <- data[i_train, ]
validating <- data[-i_train, ]
```

I'll apply several Linear Discriminant Analysis to problem.
(Preprocess data first)


```r
preprocess <- function(data) {
    for (i in 1:dim(data)[2]) {
        if (is.numeric(data[, i])) {
            data[, i] <- (data[, i] - mean(data[, i])) / sd(data[, i])
        }
    }
    data
}
training <- preprocess(training)
validating <- preprocess(validating)
testing <- preprocess(testing)
```

### Linear Discriminant Analysis


```r
fit_lda <- train(classe ~ ., data=training, method="lda")
pred_lda <- predict(fit_lda, validating)
confusionMatrix(pred_lda, validating$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1116    0    0    0    0
##          B    0  759    0    0    0
##          C    0    0  684    0    0
##          D    0    0    0  643    0
##          E    0    0    0    0  721
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9991, 1)
##     No Information Rate : 0.2845     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

Now I apply trained algorithm to testing ("unseen") data:


```r
answers <- predict(fit_lda, testing)
print(answers)
```

```
##  [1] A A A B B B B B B B B C C C C D E E E E
## Levels: A B C D E
```
