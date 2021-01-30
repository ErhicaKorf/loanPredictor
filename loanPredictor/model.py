#%%
data_final_vars=data_final.columns.values.tolist()
y_train=data_final['good_bad_num']
# data_final = data_final.drop(['good_bad_flag'], axis=1)

X_train=data_final.loc[:, data_final.columns != 'good_bad_num']

#%%
y.dtype
X.dtype
X['birthdate'].values
X['customerid'].values

# %%
logit_model=sm.Logit(np.asarray(y),np.asarray(X))
result=logit_model.fit()
print(result.summary2())

# %%
y_train = data_final['good_bad_num']
X_train = data_final.loc[:, data_final.columns != 'good_bad_num']
X_train = X.drop(['customerid'], axis=1)
X_train = X.drop(['birthdate'], axis=1)



LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr').fit(X, y)
LR.predict(X)
round(LR.score(X,y), 4)
# %%
### Support Vector Machine
SVM = svm.LinearSVC()
SVM.fit(X, y)
SVM.predict(X)
round(SVM.score(X,y), 4)


# %%
# Remove two columns name is 'C' and 'D' 
X_train = X_train.drop(['customerid','birthdate','approveddate_x', 'creationdate_x','approveddate_y',
'creationdate_y','closeddate','firstduedate','firstrepaiddate',
'referredby_y','referredby_x'], axis = 1) 

#%%
# LOGISTIC REGRESSION
logreg = LogisticRegression()
logreg.fit(X_train.iloc[:10000,], y_train.iloc[:10000,])

y_test = y_train.iloc[10000:,]
x_test = X_train.iloc[10000:,]

print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(logreg.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format(logreg.score(x_test, y_test)))


#%%
# DECISION TREE
clf = DecisionTreeClassifier().fit(X_train, y_train)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
     .format(clf.score(x_test, y_test)))

# %%
# K NEAREST NEIGHBOURS
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(x_test, y_test)))


# %%
# LINEAR DISCRIMINANT ANALYSIS
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
print('Accuracy of LDA classifier on training set: {:.2f}'
     .format(lda.score(X_train, y_train)))
print('Accuracy of LDA classifier on test set: {:.2f}'
     .format(lda.score(x_test, y_test)))


# %%
# GAUSSIAN
gnb = GaussianNB()
gnb.fit(X_train, y_train)
print('Accuracy of GNB classifier on training set: {:.2f}'
     .format(gnb.score(X_train, y_train)))
print('Accuracy of GNB classifier on test set: {:.2f}'
     .format(gnb.score(x_test, y_test)))


# %%
 # SVM 
svm = SVC()
svm.fit(X_train, y_train)
print('Accuracy of SVM classifier on training set: {:.2f}'
     .format(svm.score(X_train, y_train)))
print('Accuracy of SVM classifier on test set: {:.2f}'
     .format(svm.score(x_test, y_test)))


# %%
pred = clf.predict(x_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
# %%
pred = knn.predict(x_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
# %%
