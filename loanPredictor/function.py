#%%
def loanPredictor(df_test_demo,df_train_demo,df_test_perf,df_train_perf,df_test_prev,df_train_prev):
    # compile the list of dataframes you want to merge
    data_frames = [df_test_demo, df_test_perf, df_test_prev]
    # merging the data sets
    df_test_merged = reduce(lambda  left,right: pd.merge(left,right,on=['customerid'],
                                                how='inner'), data_frames)

    # compile the list of dataframes you want to merge
    data_frames_train = [df_train_demo, df_train_perf, df_train_prev]
    # merging the data sets
    df_train_merged = reduce(lambda  left,right: pd.merge(left,right,on=['customerid'],
                                                how='inner'), data_frames_train)

    # Creating dummy variables for the categorical variables
    cat_vars=['bank_account_type','bank_name_clients','employment_status_clients','level_of_education_clients'
                    ,'bank_branch_clients']
    for var in cat_vars:
        cat_list='var'+'_'+var
        # print(cat_list)
        cat_list = pd.get_dummies(df_train_merged[var], prefix=var)
        data1=df_train_merged.join(cat_list)
        df_train_merged=data1

    data_vars=df_train_merged.columns.values.tolist()
    to_keep=[i for i in data_vars if i not in cat_vars]

    data_final=df_train_merged[to_keep]
    data_final.columns.values

    data_final['good_bad_num'] = np.where(data_final['good_bad_flag']== 'Good', 1, 0)

    # Creating dummy variables for the categorical variables test set
    cat_vars=['bank_account_type','bank_name_clients','employment_status_clients','level_of_education_clients'
                    ,'bank_branch_clients']
    for var in cat_vars:
        cat_list='var'+'_'+var
        # print(cat_list)
        cat_list = pd.get_dummies(df_test_merged[var], prefix=var)
        data1=df_test_merged.join(cat_list)
        df_test_merged=data1

    data_vars=df_test_merged.columns.values.tolist()
    to_keep=[i for i in data_vars if i not in cat_vars]

    data_final_test=df_test_merged[to_keep]
    data_final_test.columns.values

    y_train = data_final['good_bad_num']
    X_train = data_final.loc[:, data_final.columns != 'good_bad_num']

    X_train = X_train.drop(['customerid','birthdate','approveddate_x', 'creationdate_x','approveddate_y',
        'creationdate_y','good_bad_flag','closeddate','firstduedate','firstrepaiddate',
        'referredby_y','referredby_x'], axis = 1) 

    # LOGISTIC REGRESSION FEATURE SELECTION
    logreg = LogisticRegression()
    rfe = RFE(logreg, 20)
    rfe = rfe.fit(X_train, y_train.values.ravel())
    # print(rfe.support_)
    # print(rfe.ranking_)

    cols = ['longitude_gps','latitude_gps','systemloanid_x','loannumber_x',
    'loanamount_x', 'totaldue_x', 'termdays_x', 'systemloanid_y',
        'loannumber_y', 'loanamount_y', 'totaldue_y', 'termdays_y',
        'bank_account_type_Other',
        'bank_account_type_Savings','bank_name_clients_Diamond Bank', 'bank_name_clients_EcoBank',
        'bank_name_clients_Stanbic IBTC','employment_status_clients_Permanent',
        'employment_status_clients_Self-Employed','level_of_education_clients_Graduate']

    X_train = X_train[cols]

    # Breaking training set into smaller training and test set. another train and validation set
    x_train,x_test,y_train,y_test=train_test_split(X_train,y_train,test_size=0.2)     

    # LOGISTIC REGRESSION
    logreg = LogisticRegression()
    logreg.fit(x_train,y_train)


    print('Accuracy of Logistic regression classifier on training set: {:.2f}'
        .format(logreg.score(x_train, y_train)))
    print('Accuracy of Logistic regression classifier on test set: {:.2f}'
        .format(logreg.score(x_test, y_test)))

    # DECISION TREE
    clf = DecisionTreeClassifier().fit(x_train, y_train)
    print('Accuracy of Decision Tree classifier on training set: {:.2f}'
        .format(clf.score(x_train, y_train)))
    print('Accuracy of Decision Tree classifier on test set: {:.2f}'
        .format(clf.score(x_test, y_test)))

    # K NEAREST NEIGHBOURS
    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train)
    print('Accuracy of K-NN classifier on training set: {:.2f}'
        .format(knn.score(x_train, y_train)))
    print('Accuracy of K-NN classifier on test set: {:.2f}'
        .format(knn.score(x_test, y_test)))

    # LINEAR DISCRIMINANT ANALYSIS
    lda = LinearDiscriminantAnalysis()
    lda.fit(x_train, y_train)
    print('Accuracy of LDA classifier on training set: {:.2f}'
        .format(lda.score(x_train, y_train)))
    print('Accuracy of LDA classifier on test set: {:.2f}'
        .format(lda.score(x_test, y_test)))

    # GAUSSIAN
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    print('Accuracy of GNB classifier on training set: {:.2f}'
        .format(gnb.score(x_train, y_train)))
    print('Accuracy of GNB classifier on test set: {:.2f}'
        .format(gnb.score(x_test, y_test)))

    # SVM 
    svm = SVC()
    svm.fit(x_train, y_train)
    print('Accuracy of SVM classifier on training set: {:.2f}'
        .format(svm.score(x_train, y_train)))
    print('Accuracy of SVM classifier on test set: {:.2f}'
        .format(svm.score(x_test, y_test)))

    # TESTING THE PREDICTIONS FOR TEST SET DECISION CLASSIFIER
    pred = clf.predict(x_test)
    print('Decision tree confusion matrix',confusion_matrix(y_test, pred))
    # print(classification_report(y_test, pred))

    # TESTING THE PREDICTIONS FOR TEST SET KNN
    pred = knn.predict(x_test)
    print('KNN confusion matrix',confusion_matrix(y_test, pred))
    # print(classification_report(y_test, pred))

    # PREDICT LOAN GOOD BAD FLAG ON REAL TEST DATA
    df_test_merged_pred = df_test_merged[cols]
    pred = clf.predict(df_test_merged_pred)

    # Add predictions to test data set
    df_test_merged_pred['Good_Bad_Flag'] = pred

    # Move to second column
    good_bad_col = df_test_merged_pred.pop('Good_Bad_Flag')
    customerid_col = df_test_merged.pop('customerid')
    df_test_merged_pred.insert(0, 'customerid', customerid_col)
    df_test_merged_pred.insert(1, 'Good_Bad_Flag', good_bad_col)

    print(df_test_merged_pred[['customerid','Good_Bad_Flag']])
# %%
loanPredictor(df_test_demo,df_train_demo,df_test_perf,df_train_perf,df_test_prev,df_train_prev)
# %%
