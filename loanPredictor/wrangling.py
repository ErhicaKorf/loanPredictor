#%%
# %%
# Creating dummy variables for the categorical variables
cat_vars=['bank_account_type','bank_name_clients','employment_status_clients','level_of_education_clients'
                ,'good_bad_flag','bank_branch_clients']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(df_train_merged[var], prefix=var)
    data1=df_train_merged.join(cat_list)
    data=data1

data_vars=data.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]

data_final=data[to_keep]
data_final.columns.values

#%%
# sample the bad loans using SMOTE algorithm. Create synthetic samples of minor
# class (bad loans) and not copies using k-nearest neighbours 