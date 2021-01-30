#%%
# Reading in the data 
df_test_demo = pd.read_csv('C:/Users/erhicakorf/Documents/GitHub/loanPredictor/testdemographics.csv')
df_train_demo = pd.read_csv('C:/Users/erhicakorf/Documents/GitHub/loanPredictor/traindemographics.csv')
df_test_perf = pd.read_csv('C:/Users/erhicakorf/Documents/GitHub/loanPredictor/testperf.csv')
df_train_perf = pd.read_csv('C:/Users/erhicakorf/Documents/GitHub/loanPredictor/trainperf.csv')
df_test_prev = pd.read_csv('C:/Users/erhicakorf/Documents/GitHub/loanPredictor/testprevloans.csv')
df_train_prev = pd.read_csv('C:/Users/erhicakorf/Documents/GitHub/loanPredictor/trainprevloans.csv')

#%%
print(df_test_demo.shape)
print(df_train_demo.shape)
print(df_test_perf.shape)
print(df_train_perf.shape)
print(df_test_prev.shape)
print(df_train_prev.shape)

# %%
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

# %%
# Check for missing values
df_train_merged.isnull().sum()
# %%
# Visualisations 

# sns.countplot(x='good_bad_flag',data=df_train_merged,palette='hls')
# pd.crosstab(df_train_merged.birthdate,df_train_merged.good_bad_flag).plot(kind='bar')
# table=pd.crosstab(df_train_merged.loanamount_x,df_train_merged.good_bad_flag)
# table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
# df_train_merged.level_of_education_clients.hist()
# %%
df_train_merged.groupby('good_bad_flag').mean()
# %%
df_train_merged.groupby('level_of_education_clients').mean()


# %%
df_train_merged['good_bad_flag'][(df_train_merged['employment_status_clients']=='Permanent')&(df_train_merged['good_bad_flag']=='Bad')]

#%%
df_train_merged['good_bad_flag'][df_train_merged['employment_status_clients']=='Permanent']
# %%
