# machine_learning assignment
GGraner  
29 Oct 2016  

## Synopsis
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. 

### Project goal
The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases. 

## Data Documentation
* Data and more Information for this project come from this Website: http://groupware.les.inf.puc-rio.br/har, proposing a dataset with 5 classes (sitting-down, standing-up, standing, walking, and sitting) collected on 8 hours of activities of 4 healthy subjects.
* Training data: [pml-training.csv](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv)
* Test data: [pml-testing.csv](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv) 

## Data Processing

### Downloading and reading the data

```r
        setwd("~/data_science/machine_learning")
        # downloading data
        fileUrltest<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
        if(!file.exists("./pml-testing.csv")) {
                cat("downloading test data...this may take some time.\n")
                download.file(fileUrltest, destfile="./pml-testing.csv", method="curl")
        }

        fileUrltraining<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
        if(!file.exists("./pml-training.csv")) {
                cat("downloading training data...this may take some time.\n")
                download.file(fileUrltraining, destfile="./pml-training.csv", method="curl")
        }
        
        # reading data
        if("pml-training.csv" %in% dir() & "pml-testing.csv" %in% dir()){
                cat("reading data...this may take some time.\n")
                test_data <- read.csv("pml-testing.csv",na.strings=c("NA","#DIV/0!",""))
                training_data <- read.csv("pml-training.csv",na.strings=c("NA","#DIV/0!",""))
        }else return("relevant file is missing, please check")
```

```
## reading data...this may take some time.
```

### Preprocessing the data

```r
        library(dplyr)
        library(caret)
        library(randomForest)
        library(rpart)
        library(rpart.plot)
        library(rattle)

        # These variables are irrelevant for the analyis: user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, new_window, num_window (col 1:7)
        training_data   <- training_data[,-c(1:7)]

        
        # Removing near Zero variance variables due lack of influence
        near_zero_var <- nearZeroVar(training_data)
        training_data <- training_data[,-near_zero_var]

        # only keep columns with at least 10% NAs
        training_data <- training_data[, colSums(is.na(training_data)) < nrow(training_data) * 0.1]
        
        # partition the data: 75% ... training,  25% ... testing
        set.seed(998)
        train_set <- createDataPartition(y=training_data$classe, p=0.75, list=FALSE)
        train_set_data <- training_data[train_set, ] 
        test_set_data <- training_data[-train_set, ]
```
### Preprocessed dataset        

```r
        dim(train_set_data )
```

```
## [1] 14718    53
```

```r
        paste("The preprocessed training dataset has 53 variables. These variables are: ")
```

```
## [1] "The preprocessed training dataset has 53 variables. These variables are: "
```

```r
        names(train_set_data)
```

```
##  [1] "roll_belt"            "pitch_belt"           "yaw_belt"            
##  [4] "total_accel_belt"     "gyros_belt_x"         "gyros_belt_y"        
##  [7] "gyros_belt_z"         "accel_belt_x"         "accel_belt_y"        
## [10] "accel_belt_z"         "magnet_belt_x"        "magnet_belt_y"       
## [13] "magnet_belt_z"        "roll_arm"             "pitch_arm"           
## [16] "yaw_arm"              "total_accel_arm"      "gyros_arm_x"         
## [19] "gyros_arm_y"          "gyros_arm_z"          "accel_arm_x"         
## [22] "accel_arm_y"          "accel_arm_z"          "magnet_arm_x"        
## [25] "magnet_arm_y"         "magnet_arm_z"         "roll_dumbbell"       
## [28] "pitch_dumbbell"       "yaw_dumbbell"         "total_accel_dumbbell"
## [31] "gyros_dumbbell_x"     "gyros_dumbbell_y"     "gyros_dumbbell_z"    
## [34] "accel_dumbbell_x"     "accel_dumbbell_y"     "accel_dumbbell_z"    
## [37] "magnet_dumbbell_x"    "magnet_dumbbell_y"    "magnet_dumbbell_z"   
## [40] "roll_forearm"         "pitch_forearm"        "yaw_forearm"         
## [43] "total_accel_forearm"  "gyros_forearm_x"      "gyros_forearm_y"     
## [46] "gyros_forearm_z"      "accel_forearm_x"      "accel_forearm_y"     
## [49] "accel_forearm_z"      "magnet_forearm_x"     "magnet_forearm_y"    
## [52] "magnet_forearm_z"     "classe"
```


## Data Models

### 1. Model: Fit a rpart model ("decision tree")

```r
        # model1 rpart        
        model_rpart <- rpart(classe ~ ., data=train_set_data, method="class")
        model_rpart
```

```
## n= 14718 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
##     1) root 14718 10533 A (0.28 0.19 0.17 0.16 0.18)  
##       2) roll_belt< 129.5 13358  9230 A (0.31 0.21 0.19 0.18 0.11)  
##         4) pitch_forearm< -34.35 1177     5 A (1 0.0042 0 0 0) *
##         5) pitch_forearm>=-34.35 12181  9225 A (0.24 0.23 0.21 0.2 0.12)  
##          10) yaw_belt>=169.5 603    68 A (0.89 0.05 0 0.053 0.01) *
##          11) yaw_belt< 169.5 11578  8765 B (0.21 0.24 0.22 0.21 0.12)  
##            22) magnet_dumbbell_z< -88.5 1533   648 A (0.58 0.28 0.048 0.072 0.022)  
##              44) magnet_dumbbell_y>=-565.5 1167   285 A (0.76 0.19 0.011 0.022 0.022) *
##              45) magnet_dumbbell_y< -565.5 366   154 B (0.0082 0.58 0.16 0.23 0.019) *
##            23) magnet_dumbbell_z>=-88.5 10045  7551 C (0.15 0.24 0.25 0.23 0.14)  
##              46) pitch_belt< -42.95 580    93 B (0.012 0.84 0.088 0.029 0.031) *
##              47) pitch_belt>=-42.95 9465  7022 C (0.16 0.2 0.26 0.24 0.14)  
##                94) magnet_dumbbell_y< 287.5 4066  2368 C (0.18 0.12 0.42 0.16 0.12)  
##                 188) magnet_belt_z< -326.5 2409  1734 C (0.25 0.13 0.28 0.19 0.15)  
##                   376) roll_belt>=123.5 369    66 A (0.82 0.046 0.035 0 0.098) *
##                   377) roll_belt< 123.5 2040  1378 C (0.15 0.14 0.32 0.22 0.16)  
##                     754) magnet_dumbbell_y>=173.5 855   570 E (0.28 0.17 0.15 0.069 0.33)  
##                      1508) roll_dumbbell< 16.49709 453   217 A (0.52 0.31 0.035 0.015 0.12)  
##                        3016) roll_arm< 86.4 248    32 A (0.87 0.0081 0.02 0 0.1) *
##                        3017) roll_arm>=86.4 205    68 B (0.098 0.67 0.054 0.034 0.15) *
##                      1509) roll_dumbbell>=16.49709 402   172 E (0.0075 0.017 0.27 0.13 0.57) *
##                     755) magnet_dumbbell_y< 173.5 1185   649 C (0.06 0.12 0.45 0.33 0.04)  
##                      1510) pitch_belt>=26.05 152    61 B (0.35 0.6 0.033 0.0066 0.013) *
##                      1511) pitch_belt< 26.05 1033   502 C (0.017 0.05 0.51 0.37 0.044)  
##                        3022) pitch_forearm< 37.65 773   278 C (0.01 0.041 0.64 0.27 0.043)  
##                          6044) magnet_belt_x< 52 633   138 C (0.013 0.051 0.78 0.13 0.024) *
##                          6045) magnet_belt_x>=52 140    18 D (0 0 0 0.87 0.13) *
##                        3023) pitch_forearm>=37.65 260    78 D (0.038 0.077 0.14 0.7 0.046) *
##                 189) magnet_belt_z>=-326.5 1657   634 C (0.075 0.11 0.62 0.13 0.069)  
##                   378) yaw_arm< -116 121     3 A (0.98 0.025 0 0 0) *
##                   379) yaw_arm>=-116 1536   513 C (0.0046 0.12 0.67 0.14 0.074) *
##                95) magnet_dumbbell_y>=287.5 5399  3804 D (0.15 0.26 0.14 0.3 0.16)  
##                 190) accel_dumbbell_y< -36.5 461    66 C (0.0022 0.069 0.86 0.035 0.037) *
##                 191) accel_dumbbell_y>=-36.5 4938  3359 D (0.16 0.28 0.071 0.32 0.17)  
##                   382) yaw_belt< -2.825 4028  2691 B (0.18 0.33 0.065 0.22 0.2)  
##                     764) roll_belt>=112.5 375     0 B (0 1 0 0 0) *
##                     765) roll_belt< 112.5 3653  2691 B (0.2 0.26 0.071 0.24 0.22)  
##                      1530) magnet_dumbbell_z< -16.5 840   361 A (0.57 0.26 0 0.15 0.018)  
##                        3060) gyros_dumbbell_y< 0.62 717   239 A (0.67 0.14 0 0.17 0.02) *
##                        3061) gyros_dumbbell_y>=0.62 123     2 B (0.0081 0.98 0 0 0.0081) *
##                      1531) magnet_dumbbell_z>=-16.5 2813  2033 E (0.093 0.26 0.093 0.27 0.28)  
##                        3062) accel_forearm_x>=-100.5 2037  1353 E (0.11 0.31 0.11 0.13 0.34)  
##                          6124) roll_belt>=-0.65 1659  1028 B (0.14 0.38 0.14 0.084 0.26)  
##                           12248) magnet_belt_z>=-331 924   496 B (0.071 0.46 0.25 0.11 0.11) *
##                           12249) magnet_belt_z< -331 735   402 E (0.22 0.28 0 0.054 0.45)  
##                             24498) roll_dumbbell< 42.55205 232    44 B (0.14 0.81 0 0.0043 0.043) *
##                             24499) roll_dumbbell>=42.55205 503   180 E (0.25 0.03 0 0.078 0.64) *
##                          6125) roll_belt< -0.65 378   126 E (0.0079 0.011 0 0.31 0.67)  
##                           12250) magnet_dumbbell_z>=246 119     0 D (0 0 0 1 0) *
##                           12251) magnet_dumbbell_z< 246 259     7 E (0.012 0.015 0 0 0.97) *
##                        3063) accel_forearm_x< -100.5 776   266 D (0.045 0.14 0.039 0.66 0.12) *
##                   383) yaw_belt>=-2.825 910   224 D (0.053 0.038 0.098 0.75 0.057) *
##       3) roll_belt>=129.5 1360    57 E (0.042 0 0 0 0.96) *
```

```r
        # predicting:
        prediction_rpart <- predict(model_rpart, test_set_data, type = "class")

        # Test results on TestTrainingSet data set:
        confusionMatrix(prediction_rpart, test_set_data$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1224  124    7   68   36
##          B   68  688  114   61   54
##          C    8   76  652  146   57
##          D   35   48   45  507   66
##          E   60   13   37   22  688
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7665          
##                  95% CI : (0.7544, 0.7783)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.7041          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8774   0.7250   0.7626   0.6306   0.7636
## Specificity            0.9330   0.9249   0.9291   0.9527   0.9670
## Pos Pred Value         0.8389   0.6985   0.6944   0.7233   0.8390
## Neg Pred Value         0.9504   0.9334   0.9488   0.9293   0.9478
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2496   0.1403   0.1330   0.1034   0.1403
## Detection Prevalence   0.2975   0.2009   0.1915   0.1429   0.1672
## Balanced Accuracy      0.9052   0.8249   0.8458   0.7916   0.8653
```


### 2. Model: Classification and Regression with Random Forest ("RF")

```r
        # model2 random forest
        model_rf <- randomForest(classe ~ ., data=train_set_data)
        model_rf
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = train_set_data) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.53%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 4182    3    0    0    0 0.0007168459
## B   15 2830    3    0    0 0.0063202247
## C    0   16 2551    0    0 0.0062329568
## D    0    0   32 2378    2 0.0140961857
## E    0    0    1    6 2699 0.0025868441
```

```r
        # predicting:
        prediction_rf <- predict(model_rf, test_set_data, type = "class")

        # Test results on TestTrainingSet data set:
        confusionMatrix(prediction_rf, test_set_data$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1394    6    0    0    0
##          B    0  942    6    0    0
##          C    0    1  847    6    0
##          D    0    0    2  798    2
##          E    1    0    0    0  899
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9951          
##                  95% CI : (0.9927, 0.9969)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9938          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9993   0.9926   0.9906   0.9925   0.9978
## Specificity            0.9983   0.9985   0.9983   0.9990   0.9998
## Pos Pred Value         0.9957   0.9937   0.9918   0.9950   0.9989
## Neg Pred Value         0.9997   0.9982   0.9980   0.9985   0.9995
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2843   0.1921   0.1727   0.1627   0.1833
## Detection Prevalence   0.2855   0.1933   0.1741   0.1635   0.1835
## Balanced Accuracy      0.9988   0.9956   0.9945   0.9958   0.9988
```

## Results
### Comparing Models
1. Model RPART ("Decision Tree"): 
+ Accuracy: 0.7665
+ expected out of sample error (1-accuracy): 0.2335

The RPART Model has an Accuracy fo 76.65 % (95% CI: 75,44; 77,83), with an out of sample error 23.35 %.

2. Model Random Forest RF: 
+ Accuracy: 0.9951
+ expected out of sample error (1-accuracy): 0.0049

The RF Model has an Accuracy fo 99.51 % (95% CI: 99,27,44; 99,69), with an out of sample error 0.49 %.

Due the Random Forest Model has a better performance, it is chosen to predict the test cases.


### Testing Model on Test Data Set
Predict 20 different test cases in relation to the test data. Model 1, Random Forest, has the highest accuracy.  

```r
# Apply your machine learning algorithm to the 20 test cases available in the test data above and submit your predictions in appropriate format to the Course Project Prediction Quiz for automated grading.
prediction_test <- predict(model_rf, newdata = test_data , type="class")

print(prediction_test)
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

```r
write.csv(prediction_test, "submit.csv")
```

## Appendix 

### Plots

```r
# Plot of the rpart model
rpart.plot(model_rpart, type=1, main="Classification Tree")
```

![](machine_learning_files/figure-html/plots-1.png)

```r
# Plot error level of RF Model
plot(model_rf, main="2. Model: Classification and Regression with Random Forest")
```

![](machine_learning_files/figure-html/plots-2.png)

## Session Info
* R version 3.2.3 (2015-12-10)
* Platform: x86_64-apple-darwin13.4.0 (64-bit)
* Running under: OS X 10.10 (Yosemite)
* locale: [1] en_US.UTF-8/en_US.UTF-8/en_US.UTF-8/C/en_US.UTF-8/en_US.UTF-8
