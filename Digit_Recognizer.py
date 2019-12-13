from matplotlib import pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import csv

dataset = pd.read_csv("C:/Users/Rahul/Downloads/Train.csv")
dr = pd.read_csv("C:/Users/Rahul/Downloads/test.csv")

#print(dataset)

clf = DecisionTreeClassifier()

#training
xtrain = dataset.iloc[0:42000, 1:].values
train_label = dataset.iloc[0:42000, 0].values

clf.fit(xtrain, train_label)

#testing
xtest = dr.iloc[0:28000, :].values
actual_label = dr.iloc[0:28000, 0].values

#accuracy

p = clf.predict(xtest)
#count = 0

f = open('C:/Users/Rahul/Downloads/DRanswer.csv', 'w', newline='')
cs = csv.writer(f)
cs.writerow(['ImageId', 'Label'])

for i in range(0,28000):

        d = xtest[i]
        d.shape = (28, 28)
        plt.imshow(255-d, cmap="gray")
        cs.writerow([i+1, clf.predict([xtest[i]])])

