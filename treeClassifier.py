from sklearn import tree

# training data
labels = ["apple","apple","orange","orange"]
features = [[140,1],[130,1],[150,0],[170,0]]

# classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features,labels)
print(clf.predict([[169,1]]))