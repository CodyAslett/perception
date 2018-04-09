library(arules)
library(arulesViz)
library(datasets)

# col_names avoid the first row from becoming the header for the csv file
#groceries <- read_csv("Desktop/Python/W18/CS450/W09/groceries.csv", col_names=FALSE)
data(Groceries)

cat('Apriori Algorithm BYU-Idaho\n')
cat('-----------------------------\n')
cat('Note: Groceries dataset will be utilized for this algorithm.\n\n')

cat('Sort by:\n')
cat('1 - Support\n')
cat('2 - Confidence\n')
cat('3 - Lift\n')
sortNum <- readline('>>> ')

cat('Output type:\n')
cat('1 - Table\n')
cat('2 - Interactive Graph\n')
typeNum <- readline('>>> ')

if (typeNum == 1) {
    cat('Number of outputs\n')
    outputNum <- readline('>>> ')
}

rules <- apriori(Groceries, parameter = list(supp = .001, conf = 0.8))

if (sortNum == 1) {
    rules <- sort(rules, by="support", decreasing=TRUE)
} else if (sortNum == 2) {
    rules <- sort(rules, by="confidence", decreasing=TRUE)
} else {
    rules <- sort(rules, by="lift", decreasing=TRUE)
}

if (typeNum == 1) {
    inspect(rules[1:outputNum])
} else {
    plot(rules[1:3],method="graph",interactive=TRUE,shading=NA)
}