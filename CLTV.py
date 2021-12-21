#pip install lifetimes
#pip install sqlalchemy
#conda install -c anaconda mysql-connector-python
#conda install -c conda-forge mysql
#conda upgrade conda

from sqlalchemy import create_engine
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler

df_ = pd.read_excel("Dataset/online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")
df = df_.copy()
df.shape

creds = {'user': 'group_3',
         'passwd': 'miuul',
         'host': '34.79.73.237',
         'port': 3306,
         'db': 'group_3'}

connstr = 'mysql+mysqlconnector://{user}:{passwd}@{host}:{port}/{db}'

# sqlalchemy engine for MySQL connection.
conn = create_engine(connstr.format(**creds))

# conn.close()
pd.read_sql_query("show databases;", conn)
pd.read_sql_query("show tables", conn)



pd.read_sql_query("select * from online_retail_2010_2011 limit 10", conn)

retail_mysql_df = pd.read_sql_query("select * from online_retail_2010_2011", conn)


retail_mysql_df.shape
retail_mysql_df.head()
retail_mysql_df.info()
# df = retail_mysql_df.copy()


# Pre-Processing
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

df.describe().T
df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]
df.head()
df.columns
df = df[df["Country"] == "United Kingdom"]
df["TotalPrice"] = df["Quantity"] * df["Price"]
today_date = dt.datetime(2011, 12, 11)
cltv_df = df.groupby('Customer ID').agg({'InvoiceDate': [lambda date: (date.max() - date.min()).days,
                                                         lambda date: (today_date - date.min()).days],
                                         'Invoice': lambda num: num.nunique(),
                                         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})
cltv_df.head()
cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']
#T= kendi içerisinde ilk alışveriş tarihiyle son alışveriş tarihi arasıdnaki geçen süre.

cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]
cltv_df = cltv_df[(cltv_df['frequency'] > 1)]
cltv_df = cltv_df[cltv_df["monetary"] > 0]
cltv_df["recency"] = cltv_df["recency"] / 7
cltv_df["T"] = cltv_df["T"] / 7
#T= the elapsed time between the first shopping date and the last shopping date in itself.

# TASK 1
# 6 Month CLTV Prediction

#BG-NBD Model
bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])


# GAMMA-GAMMA Model
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary'])


cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=6,  # 6 month
                                   freq="W",
                                   discount_rate=0.01)

cltv.head()  #6 month cltv (seri)
cltv.describe().T



cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
cltv_final.sort_values(by="clv", ascending=False).head(10)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(cltv_final[["clv"]])
cltv_final["scaled_clv"] = scaler.transform(cltv_final[["clv"]])
cltv_final.sort_values(by="scaled_clv", ascending=False).head()
#scaled on cltv df

# TASK 2
# Calculate 1-month and 12-month CLTV for 2010-2011 UK customers.
# Analyze the top 10 people at 1 month CLTV and the 10 highest at 12 months.

bgf.predict(4,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sort_values(ascending=False).head(10)


cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])
cltv_bir_on = cltv_df.sort_values("expected_purc_1_month", ascending=False).head(10)

#Top 10 people at 12 months
bgf.predict(4*12,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sort_values(ascending=False).head(10)


cltv_df["expected_purc_12_month"] = bgf.predict(4*12,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])
cltv_bir_oniki = cltv_df.sort_values("expected_purc_12_month", ascending=False).head(10)



# The first ten customers in the demand forecast for one month and the customer forecast for the twelve months are not the same.
#12 months also includes people with a lower frequency of shopping.


# TASK 3
# For 2010-2011 UK customers all your customers into 4 groups based on 6-month CLTV.
# (segment) and add the group names to the dataset.
# Make short 6-month action suggestions to the management for 2 groups you will choose from among 4 groups.

cltv.head()  #6 month CLTV
cltv_final["segment"] = pd.qcut(cltv_final["scaled_clv"], 4, labels=["D", "C", "B", "A"])

cltv_final.head()
cltv_final.sort_values(by="scaled_clv", ascending=False).head(50)

cltv_final = cltv_final.reset_index()
cltv_final.groupby("segment").agg(
    {"count", "mean", "sum"})


cltv_final[cltv_final["segment"] == "A"]
cltv_final.groupby("segment").agg(
    {"count", "mean", "sum"})


A = cltv_final.loc[cltv_final["segment"] =="A" ,"segment"]
C = cltv_final.loc[cltv_final["segment"] =="C" ,"segment"]


# TASK 4
# extract to excel
cltv_df.to_sql(name='zeynep_kaya', con=conn, if_exists='replace', index=False)
pd.read_sql_query("show tables", conn)

