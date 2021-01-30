#%%
# %%
# Creating dummy variables for the categorical variables
cat_vars=['bank_account_type','bank_name_clients','employment_status_clients','level_of_education_clients'
                ,'bank_branch_clients']
for var in cat_vars:
    cat_list='var'+'_'+var
    print(cat_list)
    cat_list = pd.get_dummies(df_train_merged[var], prefix=var)
    data1=df_train_merged.join(cat_list)
    df_train_merged=data1

data_vars=df_train_merged.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]

data_final=df_train_merged[to_keep]
data_final.columns.values

#%%
data_final['good_bad_num'] = np.where(data_final['good_bad_flag']== 'Good', 1, 0)

# %%
# Doing the same for the test data
# Creating dummy variables for the categorical variables
cat_vars=['bank_account_type','bank_name_clients','employment_status_clients','level_of_education_clients'
                ,'bank_branch_clients']
for var in cat_vars:
    cat_list='var'+'_'+var
    print(cat_list)
    cat_list = pd.get_dummies(df_test_merged[var], prefix=var)
    data1=df_test_merged.join(cat_list)
    df_test_merged=data1

data_vars=df_test_merged.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]

data_final_test=df_test_merged[to_keep]
data_final_test.columns.values

# %%
# %%
