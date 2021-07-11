from sklearn.feature_extraction.text import TfidfVectorizer
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import urllib.parse
import matplotlib.pyplot as plt
import pickle
from sklearn.svm import SVC
def loadFile(name):
    directory = str(os.getcwd())  
    filepath = os.path.join(directory, name)  
    with open(filepath,'r', encoding="utf8") as f:  #now open the file in reading mode
        data = f.readlines() 
    data = list(set(data))
    result = []
    for d in data:
        d = str(urllib.parse.unquote(d))   #converting url encoded data to simple string
        result.append(d)
    return result
badQueries = loadFile('badqueries.txt')
validQueries = loadFile('goodqueries.txt')

badQueries = list(set(badQueries))
validQueries = list(set(validQueries))
allQueries = badQueries + validQueries
yBad = [1 for i in range(0, len(badQueries))]  #labels, 1 for malicious and 0 for clean
yGood = [0 for i in range(0, len(validQueries))]
y = yBad + yGood
queries = allQueries

vectorizer = TfidfVectorizer(min_df = 0.0, analyzer="char", sublinear_tf=True, ngram_range=(1,3)) #converting data to vectors
X = vectorizer.fit_transform(queries)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #splitting data

badCount = len(badQueries)
validCount = len(validQueries)
svclassifier = SVC(kernel='sigmoid',probability=True)
svclassifier.fit(X_train, y_train)
predicted = svclassifier.predict(X_test)

#pickle.dump(lgs, open('p1.pkl','wb')) #to connect with frontend
#pickle.dump(vectorizer, open('p2.pkl','wb')) #to connect with flask
fpr, tpr, _ = metrics.roc_curve(y_test, (svclassifier.predict_proba(X_test)[:, 1]))
auc = metrics.auc(fpr, tpr)

print("Bad samples: %d" % badCount)
print("Good samples: %d" % validCount)
print("Baseline Constant negative: %.6f" % (validCount / (validCount + badCount)))
print("------------")
print("Accuracy of test dataset: %f" % svclassifier.score(X_test, y_test))  #checking the accuracy
print("Accuracy of train dataset: %f" % svclassifier.score(X_train, y_train))
print("Precision: %f" % metrics.precision_score(y_test, predicted))
print("Recall: %f" % metrics.recall_score(y_test, predicted))
print("F1-Score: %f" % metrics.f1_score(y_test, predicted))
print("AUC: %f" % auc)

