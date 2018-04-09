# Include the LIBSVM package
library (e1071)
library(readr)

print("SVM Algorithm BYU-Idaho")
print("-----------------------")
print("1 - Letters dataset")
print("2 - Vowel dataset")

dataset <- readline("> ")

gamma <- c(0.001, 0.05, 0.01, 0.5, 0.1, 1)
letterHighestGamma <- -1
letterHighestCost <- -1
letterHighestAccuracy <- -1
letterHighestPrediction <- -1

vowelHighestGamma <- 1
vowelHighestCost <- 1
vowelHighestAccuracy <- -1
vowelHighestPrediction <- -1

if (dataset == 1) {
    cat("--------------LETTERS DATASET-------------\n")

    fileName <- "Desktop/Python/W18/CS450/W07/letters.csv"
    data <- read.csv(fileName, head=TRUE, sep=",")

    # Partition the data into training and test sets
    # by getting a random 30% of the rows as the testRows
    allRows <- 1:nrow(data)
    testRows <- sample(allRows, trunc(length(allRows) * 0.3))

    # The test set contains all the test rows
    testData <- data[testRows,]

    # The training set contains all the other rows
    trainData <- data[-testRows,]

    for (g in gamma) {
        cat("EXECUTING...\n")
        for (c in 1:2) {
            # Train an SVM model
            # Tell it the attribute to predict vs the attributes to use in the prediction,
            #  the training data to use, and the kernal to use, along with its hyperparameters.
            model <- svm(letter~., data = trainData, kernel = "radial", gamma = g, cost = c, type="C")

            # Use the model to make a prediction on the test set
            # Notice, we are not including the last column here (our target)
            prediction <- predict(model, testData[,-1])

            # Produce a confusion matrix
            confusionMatrix <- table(pred = prediction, true = testData[,1])

            # Calculate the accuracy, by checking the cases that the targets agreed
            agreement <- prediction == testData[,1]
            accuracy <- prop.table(table(agreement))
            numOfTrue <- length(which(agreement == TRUE))

            if (numOfTrue > letterHighestAccuracy) {
                letterHighestCost <- c
                letterHighestGamma <- g
                letterHighestAccuracy <- numOfTrue
                letterHighestPrediction <- accuracy
            } else {
                1
            }
        }
    }
    
    cat("Highest Cost: ", letterHighestCost, "\n")
    cat("Highest gamma: ", letterHighestGamma, "\n")
    cat("Accuracy (False True): ", letterHighestPrediction, "\n")
} else if (dataset == 2) {
    cat("--------------VOWEL DATASET-------------\n")

    fileName <- "Desktop/Python/W18/CS450/W07/vowel.csv"
    data <- read.csv(fileName, head=TRUE, sep=",")

    # Partition the data into training and test sets
    # by getting a random 30% of the rows as the testRows
    allRows <- 1:nrow(data)
    testRows <- sample(allRows, trunc(length(allRows) * 0.3))

    # The test set contains all the test rows
    testData <- data[testRows,]

    # The training set contains all the other rows
    trainData <- data[-testRows,]

    for (g in gamma) {
        cat("EXECUTING...\n")
        for (c in 1:25) {
            # Train an SVM model
            # Tell it the attribute to predict vs the attributes to use in the prediction,
            #  the training data to use, and the kernal to use, along with its hyperparameters.
            model <- svm(Class~., data = trainData, kernel = "radial", gamma = g, cost = c)

            # Use the model to make a prediction on the test set
            # Notice, we are not including the last column here (our target)
            prediction <- predict(model, testData[,-13])

            # Produce a confusion matrix
            confusionMatrix <- table(pred = prediction, true = testData[,13])

            # Calculate the accuracy, by checking the cases that the targets agreed
            agreement <- prediction == testData[,13]
            accuracy <- prop.table(table(agreement))
            numOfTrue <- length(which(agreement == TRUE))

            if (numOfTrue > vowelHighestAccuracy) {
                vowelHighestCost <- c
                vowelHighestGamma <- g
                vowelHighestAccuracy <- numOfTrue
                vowelHighestPrediction <- accuracy
            } else {
                1
            }
        }
    }
    
    cat("Highest Cost: ", vowelHighestCost, "\n")
    cat("Highest gamma: ", vowelHighestGamma, "\n")
    cat("Accuracy (False True): ", vowelHighestPrediction, "\n")
} else {
    stop("ERROR: Invalid number.")
}

