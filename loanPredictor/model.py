
#%%
y.dtype
X.dtype
X['birthdate'].values
X['customerid'].values

# %%
y_train = data_final['good_bad_num']
X_train = data_final.loc[:, data_final.columns != 'good_bad_num']

#%%
# LOGISTIC REGRESSION FEATURE SELECTION
logreg = LogisticRegression()
rfe = RFE(logreg, 20)
rfe = rfe.fit(X_train, y_train.values.ravel())
print(rfe.support_)
print(rfe.ranking_)

#%%
cols = ['longitude_gps','latitude_gps','systemloanid_x','loannumber_x',
'loanamount_x', 'totaldue_x', 'termdays_x', 'systemloanid_y',
       'loannumber_y', 'loanamount_y', 'totaldue_y', 'termdays_y',
       'bank_account_type_Other',
       'bank_account_type_Savings','bank_name_clients_Diamond Bank', 'bank_name_clients_EcoBank',
       'bank_name_clients_Stanbic IBTC','employment_status_clients_Permanent',
       'employment_status_clients_Self-Employed','level_of_education_clients_Graduate']

X_train = X_train[cols]

#%%
# Breaking training set into smaller training and test set. another train and validation set
x_train,x_test,y_train,y_test=train_test_split(X_train,y_train,test_size=0.2)     

#%%
# LOGISTIC REGRESSION
logreg = LogisticRegression()
logreg.fit(x_train,y_train)


print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(logreg.score(x_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format(logreg.score(x_test, y_test)))


#%%
# DECISION TREE
clf = DecisionTreeClassifier().fit(x_train, y_train)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(clf.score(x_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
     .format(clf.score(x_test, y_test)))

# %%
# K NEAREST NEIGHBOURS
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(x_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(x_test, y_test)))


# %%
# LINEAR DISCRIMINANT ANALYSIS
lda = LinearDiscriminantAnalysis()
lda.fit(x_train, y_train)
print('Accuracy of LDA classifier on training set: {:.2f}'
     .format(lda.score(x_train, y_train)))
print('Accuracy of LDA classifier on test set: {:.2f}'
     .format(lda.score(x_test, y_test)))


# %%
# GAUSSIAN
gnb = GaussianNB()
gnb.fit(x_train, y_train)
print('Accuracy of GNB classifier on training set: {:.2f}'
     .format(gnb.score(x_train, y_train)))
print('Accuracy of GNB classifier on test set: {:.2f}'
     .format(gnb.score(x_test, y_test)))


# %%
 # SVM 
svm = SVC()
svm.fit(x_train, y_train)
print('Accuracy of SVM classifier on training set: {:.2f}'
     .format(svm.score(x_train, y_train)))
print('Accuracy of SVM classifier on test set: {:.2f}'
     .format(svm.score(x_test, y_test)))


# %%
# TESTING THE PREDICTIONS FOR TEST SET DECISION CLASSIFIER
pred = clf.predict(x_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

# %%
# TESTING THE PREDICTIONS FOR TEST SET KNN
pred = knn.predict(x_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

# %%
# PREDICT LOAN GOOD BAD FLAG ON REAL TEST DATA
df_test_merged = df_test_merged[cols]
pred = clf.predict(df_test_merged)

# Add predictions to test data set
df_test_merged['Good_Bad_Flag'] = pred

# Move to second column
good_bad_col = df_test_merged.pop('Good_Bad_Flag')
df_test_merged.insert(1, 'Good_Bad_Flag', good_bad_col)
# %%
