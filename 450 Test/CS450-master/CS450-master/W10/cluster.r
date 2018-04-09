# Author: Sam McGrath
# Professor: Br. Burton
# Class: CS 450

library(cluster)
library(datasets)
library(readr)

cat("Clustering Algorithm BYU-Idaho\n")
cat("-----------------------\n")
cat("1 - Agglomerative Hierarchical Clustering\n")
cat("2 - K-Means Clustering\n")

algorithm <- readline("> ")

# AGGLOMERATIVE HIERARCHICAL CLUSTERING
if (algorithm == 1) {
    cat("\n1 - Produce a dendogram though normal clustering\n")
    cat("2 - Normalize dataset clustering\n")
    cat("3 - No area clustering\n")
    cat("4 - Only Frost clustering\n")

    agglo <- readline("> ")

    # Load the dataset
    data <- state.x77
    if (agglo == 1) {

        # first compute a distance matrix then cluster it
        hcOriginal <- hclust(dist(as.matrix(data)))

        # finally, plot the dendrogram
        plot(hcOriginal, xlab="States", ylab="Cluster")
    } else if (agglo == 2) {
        # Repeat the previous item with a normalized dataset and note any differences
        dataScaled <- scale(state.x77)
        hcNormalized <- hclust(dist(as.matrix(dataScaled)))

        plot(hcNormalized, xlab="States", ylab="Cluster")
    } else if (agglo == 3) {
        # Cluster with no area column
        dataNoArea <- subset(data, select = -c(Area) )
        hcNoArea <- hclust(dist(as.matrix(dataNoArea)))

        plot(hcNoArea, xlab="States", ylab="Cluster")
    } else {
        # Cluster with only frost data from state.x77
        frostData <- data[,7, drop=FALSE]
        hcFrost <- hclust(dist(as.matrix(frostData)))

        plot(hcFrost, xlab="States", ylab="Cluster")
    }
} else {
    cat("\n1 - Cluster Plot with 3 Clusters\n")
    cat("2 - Elbow Method\n")
    cat("3 - 7 Clusters Plot\n")
    cat("4 - Analyze Cluster Centers\n")

    cl <- readline("> ")

    if (cl == 1) {
        # Using K-Means
        data <- scale(state.x77)

        # cluster up into three clusters
        k_clusters <- kmeans(data, centers=3)

        cat("Print Cluster?\n")
        cat("1 - No\n")
        cat("2 - Yes\n")
        printSummary <- readline("> ")

        #determine whether to print summary
        if (printSummary == 2) {
            # summary of the clusters
            cat("Summary: \n")
            print(summary(k_clusters))
        }

        clusplot(data, k_clusters$cluster, color=TRUE, shade=TRUE, labels=2, lines=0)
    } else if (cl == 2) {
        # Using a for loop, repeat the clustering process for k = 1 to 25, and
        # plot the total within-cluster sum of squares error for each k-value.
        errorVector <- vector()
        for (k in 1:25) {
            # tot.withinss is total sum of squares across clusters
            errorVector[k] <- kmeans(data, k)$tot.withinss
        }

        #it is apparent that according to this, 7 is the best number to cluster on
        plot(errorVector, xlab="k", ylab="total within-cluster sum of squares error")
    } else if (cl == 3) {
        #according to the assignment, we must use 7 for the rest of this assignment
        k_clusters <- kmeans(data, centers=7)

        print(k_clusters$cluster)

        # Use "clusplot" to plot a 2D representation of the clustering.
        clusplot(data, k_clusters$cluster, color=TRUE, shade=TRUE, labels=2, lines=0)
    } else {
        # Analyze the centers of each of these clusters. 
        print(k_clusters$centers)
    }
}