# F-test-and-Feature-selection

There are 40 data instances and 4434 features (genes).
1st row is the class numbers
2nd - end rows contain feature vectors. Each feature vector is a column. Adding the class number at the begining,
each feature vector + label is a 4435 dimensional vector.

The f-test score and the feature number (the line/feature number) are generated.

The following tasks were performed:

Task A:

Use GenomeTrainXY.txt to select 100 top-ranked genes based on f-test.

Task B: 

Use the above selected genes as the features, train the four classifiers


a: SVM linear kernel
b: linear regression
c: KNN (k=3)
d: centroid method

Task C:

Use trained classifiers to predict the class labels of data instances provided in GenomeTestX.txt
There are total 10 data instances.
