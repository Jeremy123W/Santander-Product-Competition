#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 20:33:09 2016

@author: jeremy
"""

import pandas as pd



limit_rows   = 9000000
#limit_rows = 100000
df           = pd.read_csv("../input/train_ver2.csv",dtype={"sexo":str,
                                                    "ind_nuevo":str,
                                                    "ult_fec_cli_1t":str,
                                                    "indext":str,
                                                    }, 
                                                    nrows=limit_rows
                                                    )
unique_ids   = pd.Series(df["ncodpers"].unique())
limit_people = 1.2e4
unique_id    = unique_ids.sample(n=limit_people)
df           = df[df.ncodpers.isin(unique_id)]
    
data_summary = df.describe()


df["fecha_dato"] = pd.to_datetime(df["fecha_dato"],format="%Y-%m-%d")
df["fecha_alta"] = pd.to_datetime(df["fecha_alta"],format="%Y-%m-%d")
df["fecha_dato"].unique()

df["month"] = pd.DatetimeIndex(df["fecha_dato"]).month
df["age"]   = pd.to_numeric(df["age"], errors="coerce")

data_cleaning = df.isnull().any()


print(df.loc[(df.age >= 18) & (df.age <= 30),"age"].mean(skipna=True))
print(df.loc[(df.age >= 30) & (df.age <= 100),"age"].mean(skipna=True))
df.loc[(df.age >= 30) & (df.age <= 100),"age"].mean(skipna=True)
df.loc[df.age < 18,"age"]  = df.loc[(df.age >= 18) & (df.age <= 30),"age"].mean(skipna=True)
df.loc[df.age > 100,"age"] = df.loc[(df.age >= 30) & (df.age <= 80),"age"].mean(skipna=True)
df["age"].fillna(df["age"].mean(),inplace=True)
df["age"]                  = df["age"].astype(int)



df.loc[df.nomprov=="CORU\xc3\x91A, A","nomprov"] = "CORUNA, A"
df.loc[df.nomprov.isnull(),"nomprov"] = "UNKNOWN"



grouped        = df.groupby("nomprov").agg({"renta":lambda x: x.median(skipna=True)}).reset_index()
new_incomes    = pd.merge(df,grouped,how="inner",on="nomprov").loc[:, ["nomprov","renta_y"]]
new_incomes    = new_incomes.rename(columns={"renta_y":"renta"}).sort_values("renta").sort_values("nomprov")
df.sort_values("nomprov",inplace=True)
df             = df.reset_index()
new_incomes    = new_incomes.reset_index()
df_rent_not_null = df.loc[df.renta.notnull()]
raw_rent_by_age = df_rent_not_null.groupby("age")['renta'].median()
rent_by_age=df_rent_not_null.groupby([pd.cut(df_rent_not_null["age"], [0,29,40,pd.np.inf], right=False),"nomprov"]).median()
rent_by_age.to_csv("rent_by_age_and_nomprov.csv")

