
stockdata=pd.read_csv('dji.csv', encoding = "ISO-8859-1")

stockdata = stockdata.drop(stockdata.index[277])
stockdata = stockdata.drop(stockdata.index[348])
stockdata = stockdata.drop(stockdata.index[681])

# Removed those nan datesbefore adding news headlines column.

#Merge both news and stock dataset on basis of date
# size matches with original df

stockdata[stockdata['Close']!= stockdata['Adj Close']]
#this shows close and adj close are same for all rows
# drop adj close

stockdata.drop(['Adj Close'],axis=1,inplace=True)


stockdata['headlines'] = headlines_main
# Now we have dataset of stocks+news
stockdata['sentiment']=predictions_wholedataset  
# sentiment column added

sns.countplot(x='sentiment',data=stockdata)

## Stock data EDA
stockdata.head()
print(stockdata.shape)
print(stockdata.columns) 
print(stockdata.dtypes)  
desc = stockdata.describe().transpose()

stockdata.isnull().sum() 
# No null values in any column

stockdata.info()

#volume is in int-> change to float
stockdata['Volume'] = stockdata['Volume'].astype(float)
stockdata.info()

#----------------------------------------------------------------------
#creation of derived variables

# date is of type object. We change it to date time type
stockdata['Date'] = pd.to_datetime(stockdata['Date'])

stockdata.info()
stockdata['Year'] = stockdata['Date'].dt.year
stockdata['Month'] = stockdata['Date'].dt.month
stockdata['Day'] = stockdata['Date'].dt.day
stockdata['Quarter'] = stockdata['Date'].dt.quarter
# 3 months form a quarter

stockdata['semester'] = np.where(stockdata['Quarter'].isin([1,2]),1,2)
#quarter1,2(6 months) is sem1 , later is sem2

stockdata['Dayofweek'] = stockdata['Date'].dt.dayofweek
stockdata['Dayofweek_name'] = stockdata['Date'].dt.day_name()
#Monday is given 0
stockdata['Dayofyear'] = stockdata['Date'].dt.dayofyear
stockdata['Weekofyear'] = stockdata['Date'].dt.weekofyear


#_---------------------------------------------------------------------

# making yesterday columns for close and adj close to fight 100% corr. Also we need this step since there is no use of our model to predict closing price at day end when we will
# already be having actual close at day end. So we want our model to predict on the basis of past data. Thus yesterday columns will act as indep variables. This way we can use our
# model at any time of day, not necessarily day end since we will have open,max,low price at any point of day along with  previous day close and adj close. We can predict todays  close and adj close

stockdata.corr()['Close']
stockdata['yesterday_open'] = stockdata['Open'].shift()
stockdata['yesterday_close'] = stockdata['Close'].shift()
stockdata['yesterday_high'] = stockdata['High'].shift()
stockdata['yesterday_low'] = stockdata['Low'].shift()

stockdata = stockdata.fillna(method='bfill')
#------------------------------------------------

stockdata.columns
cols = ['Date', 'Year', 'Month','Day', 'Dayofweek','Dayofweek_name'
        ,'Dayofyear','Weekofyear',
       'Quarter', 'semester','headlines', 'sentiment',
       'Open','yesterday_open', 'High','yesterday_high',
       'Low','yesterday_low', 'Close','yesterday_close',
       'Volume'
       ]
#Rearranging columns for simplicity
stockdata = stockdata.reindex(columns=cols)
#---------------------------------------------------------
#get economic data

from pandas_datareader import data
start='2008/08/08'
end='2016/06/30'
dataset=stockdata
qstart='2008/07/01'
qend='2016/07/01'
mstart='2008/08/01'
mend='2016/07/01'
ystart='2008/01/01'
yend='2017/01/01'
wstart='2008/07/10'
wend='2016/07/10'

# gdp RELEASED on 1st day of every quarter. So we adjust dates  so that we get data corresponding to stockdata. Then we convert quarterly freq to daily freq by appending value at before date  to all the dates until next quarter results comes.
# our date- 2018/08/08->2016/06/30
#gdp(quarterly frequency)

def economic(start,end,dataset):
    quarterly = data.DataReader(['GDP','MSPUS'], 'fred', start=qstart, end=qend)
    quarterly = quarterly.asfreq('D',method='ffill')
    
    # change index as numeric and make date index as a column
    quarterly = quarterly.reset_index()
    
    # to merge with stockdata, we need to remove weekends and extra dates
    
    quarterly['Dayofweekname'] = quarterly['DATE'].dt.day_name()
    quarterly['Dayofweekname'] = quarterly['Dayofweekname'][quarterly['Dayofweekname'].apply(lambda day: day not in ['Saturday','Sunday'] )]
    
    # dropping weekend dates
    quarterly.dropna(axis=0,inplace=True)
    
    #indices gets messed up. fix them
    quarterly.reset_index(drop=True, inplace=True)
    index1 = quarterly[quarterly['DATE']==start].index.values.astype(int)[0]
    index2 = quarterly[quarterly['DATE']==end].index.values.astype(int)[0]
    
    #FETCH THESE ROWS.REMOVE OTHERS
    quarterly = quarterly.iloc[index1:index2,:]
    quarterly.reset_index(drop=True, inplace=True)
    
    # Difference is due to public holidays. We should merge now
    quarterly = quarterly.rename(columns={"DATE" : "Date",
    "MSPUS":"Housepricemedian"}) 
    quarterly.drop('Dayofweekname',axis=1,inplace=True)
    
    dataset= dataset.merge(quarterly,on='Date')
    
    # laoding interest rate,CPI,PPI,etc(Monthly frequency)
    monthly = data.DataReader(['FEDFUNDS','CPILFESL','PPIACO',
    'UNRATE','PAYEMS','RSXFS','INDPRO','UMCSENT','USSLIND','HSN1F',
    'HOUST','BUSINV','TOTBUSSMSA','RSAFS']
    , 'fred', start=mstart, end=mend)
    monthly.isnull().sum()
    monthly = monthly.asfreq('D',method='ffill')
    monthly = monthly.reset_index()
    
    # to merge with stockdata, we need to remove weekends and extra dates
    monthly['Dayofweekname'] = monthly['DATE'].dt.day_name()
    monthly['Dayofweekname'] = monthly['Dayofweekname'][monthly['Dayofweekname'].apply(lambda day: day not in ['Saturday','Sunday'] )]
    
    # dropping weekend dates
    monthly.dropna(axis=0,inplace=True)
    
    #indices gets messed up. fix them
    monthly.reset_index(drop=True, inplace=True)
    index1 = monthly[monthly['DATE']==start].index.values.astype(int)[0]
    index2 = monthly[monthly['DATE']==end].index.values.astype(int)[0]
    # FETCH THESE ROWS.REMOVE OTHERS
    monthly = monthly.iloc[index1:index2,:]
    monthly.reset_index(drop=True, inplace=True)
    monthly = monthly.rename(columns={"DATE" : "Date","CPILFESL": "CPI","PPIACO" :"PPI",
    "UNRATE":"UNEMP_RATE","PAYEMS":"CES",'RSXFS':'RETAIL_SALES_EXCLFOOD',
    "INDPRO":"IPI","UMCSENT":"CSIndex","USSLIND":"LEI",
    "HSN1F":"ONEFAMILY",
    "HOUST":'NEWHOUSEUNITS',"BUSINV":"BUSINVENTORY",
    "TOTBUSSMSA":"BUSSALES","RSAFS":"RETAIL_SALES_INCLFOOD"}) 
    monthly.drop('Dayofweekname',axis=1,inplace=True)
    dataset= dataset.merge(monthly,on='Date')
    
    # loading inflation rate( yearly frequency)
    yearly = data.DataReader(['FPCPITOTLZGUSA'], 'fred', start=ystart, end=yend)
    l2=pd.DataFrame(np.nan,columns=['FPCPITOTLZGUSA'],index=range(0,1))
    l2.index=['2020-01-01 00:00:00']
    #making datetime index
    l2 = pd.to_datetime(l2.index)
    l4=pd.DataFrame(np.nan,columns=['FPCPITOTLZGUSA'],index=l2)
    yearly = yearly.append(l4)
    yearly = yearly.fillna(method='ffill')
    yearly = yearly.asfreq('D',method='ffill')
    yearly = yearly.reset_index()
    yearly=yearly.rename(columns={'index':'DATE'})
    
    
    # to merge with stockdata, we need to remove weekends and extra dates
    yearly['Dayofweekname'] = yearly['DATE'].dt.day_name()
    yearly['Dayofweekname'] = yearly['Dayofweekname'][yearly['Dayofweekname'].apply(lambda day: day not in ['Saturday','Sunday'] )]
    # dropping weekend dates
    yearly.dropna(axis=0,inplace=True)
    #indices gets messed up. fix them
    yearly.reset_index(drop=True, inplace=True)
    index1 = yearly[yearly['DATE']==start].index.values.astype(int)[0]
    index2 = yearly[yearly['DATE']==end].index.values.astype(int)[0]

       # FETCH THESE ROWS.REMOVE OTHERS
    yearly = yearly.iloc[index1:index2,:]
    #yearly.reset_index(drop=True, inplace=True)
    yearly = yearly.rename(columns={"DATE" : "Date",'FPCPITOTLZGUSA':"INFLATION"}) 
    yearly.drop('Dayofweekname',axis=1,inplace=True)
    dataset= dataset.merge(yearly,on='Date')
    
    #M2 is released weekly in billions of dollars
    weekly = data.DataReader(['M2'], 'fred', start=wstart, end=wend)
    weekly = weekly.asfreq('D',method='ffill')
    # change index as numeric and make date index as a column
    weekly = weekly.reset_index()
    # to merge with stockdata, we need to remove weekends and extra dates
    weekly['Dayofweekname'] = weekly['DATE'].dt.day_name()
    weekly['Dayofweekname'] = weekly['Dayofweekname'][weekly['Dayofweekname'].apply(lambda day: day not in ['Saturday','Sunday'] )]
    weekly.dropna(axis=0,inplace=True)
    weekly.reset_index(drop=True, inplace=True)
    index1 = weekly[weekly['DATE']==start].index.values.astype(int)[0]
    index2 = weekly[weekly['DATE']==end].index.values.astype(int)[0]

 
    weekly = weekly.iloc[index1:index2,:]
    weekly.reset_index(drop=True, inplace=True)
    weekly = weekly.rename(columns={"DATE" : "Date"}) 
    weekly.drop('Dayofweekname',axis=1,inplace=True)
    dataset = dataset.merge(weekly,on='Date')
    return dataset

stockdata= economic(start,end,dataset)

#economic variables made
#-------------------------------------------------------------

# to visualize monthly variables, we need monthly frequency data , since we cant use data typecasted in daily frequency since we will get plots having steps   
cum_month = stockdata[['Date','Month']]
new=[]
def update():
    month=8
    counter=1
    for i in range(0,len(cum_month)):
        if((cum_month.iloc[i].Month)==month):
            new.append(counter)       
        else:
            month=cum_month.iloc[i].Month
            counter=counter+1   
            new.append(counter)
    value=new[0]
    counter123=0
    for i in range(0,len(new)):
        if(counter123==0 and value==new[i]):
            counter123=counter123+1
        elif(value==new[i] and counter123!=0):
            new[i]=np.nan
        else:
            counter123=1
            value=new[i]
            
update()
length = str([i for i in range(0,len(new))])
new= pd.DataFrame(new)
new.isna()
new.fillna(method='ffill',inplace=True)
new.isna()

new.columns = ['cum_month']
new.columns
stockdata = pd.concat([stockdata, new], axis=1, sort=False)

#visualization of variables 
stockdata.columns
corr = stockdata.corr()

#gdp,houseprices calculated quarterly , since we didnt repeat above procedure

# for quarters, we check with year only
ax = sns.lineplot(x='Year',y='GDP',data=stockdata)
ax.set( ylabel='Billions of dollars')
plt.show()

# gdp has increased gradually as expected

ax = sns.lineplot(x='Year',y='Housepricemedian',data=stockdata)
ax.set( ylabel='dollars')
plt.show()
#price of housing dropped badly during stock market crash as that was  the prime reason during recession that time

#now for monthly calculated variables, use cum_month
ax = sns.lineplot(x='cum_month',y='FEDFUNDS',data=stockdata)
ax.set( ylabel='%')
plt.show()
# interest rates decreased sharply in 2009, that led to recession as people had housing loans at a very low interest

ax = sns.lineplot(x='cum_month',y='CPI',data=stockdata)
ax.set( ylabel='Index')
plt.show()
# Inflation seen in buying price for consumer

ax = sns.lineplot(x='cum_month',y='PPI',data=stockdata)
ax.set( ylabel='Index')
plt.show()
# Inflation in terms of selling price is seen to go up and down also

ax = sns.lineplot(x='cum_month',y='UNEMP_RATE',data=stockdata)
ax.set( ylabel='%')
plt.show()
# unemp rate rose to 10% in crash of 2009-10

# current employee statistics
ax = sns.lineplot(x='cum_month',y='CES',data=stockdata)
ax.set( ylabel='Thousands of persons')
plt.show()
# In 2016, of 300 mn population 144 million were employed in an organization

# Retail sales excluding food
ax = sns.lineplot(x='cum_month',y='RETAIL_SALES_EXCLFOOD',data=stockdata)
ax.set( ylabel='Millions of dollars')
plt.show()
#In 2016 , 400 BILLION was spent in retail sales excluding food industry

ax = sns.lineplot(x='cum_month',y='RETAIL_SALES_INCLFOOD',data=stockdata)
ax.set( ylabel='Millions of dollars')
plt.show()
# In 2016, 460 billion was spend in retail+food industry

# IPI
ax = sns.lineplot(x='cum_month',y='IPI',data=stockdata)
ax.set( ylabel='Index')
plt.show()
# production dipper in 2009 due to recession but has increased since

#consumer sentiment
ax = sns.lineplot(x='cum_month',y='CSIndex',data=stockdata)
ax.set( ylabel='Index (100 at 1996)')
plt.show()

#LEI predicts 6 month future growth rate
ax = sns.lineplot(x='cum_month',y='LEI',data=stockdata)
ax.set( ylabel='Percent')
plt.show()
# As seen in 2009 index becam negative since growth was expected to drop during recession

#onefamily- No of one family houses sold
ax = sns.lineplot(x='cum_month',y='ONEFAMILY',data=stockdata)
ax.set( ylabel='Thousands')
plt.show()
# No of houses decreased to 100,00 during stock market crash and subsequent years

# Construction started at houses
ax = sns.lineplot(x='cum_month',y='NEWHOUSEUNITS',data=stockdata)
ax.set( ylabel='Thousands of units')
plt.show()
# 1.2million house construction started in 2016

#Total business inventories
ax = sns.lineplot(x='cum_month',y='BUSINVENTORY',data=stockdata)
ax.set( ylabel='Millions of dollars')
plt.show()
#1800 billion worth of inventories stored in US in 2016

#total business sales
ax = sns.lineplot(x='cum_month',y='BUSSALES',data=stockdata)
ax.set( ylabel='Millions of dollars')
plt.show()
#1350 billion worth of sales was done in 2015

#inflation rate
ax = sns.lineplot(x='Year',y='INFLATION',data=stockdata)
ax.set( ylabel='%')
plt.show()
# prices dropped heavily during crash. so inflation rate decreased

#M2- money or financial assets held by US
ax = sns.lineplot(x='Date',y='M2',data=stockdata)
ax.set( ylabel='Billions of dollars')
plt.show()

# In 2016 there were 13 trillions worth assets eld by US
#-----------------------------------------------------------------------

# Visualization on sentiment variable
stockdata.groupby('sentiment')['sentiment'].value_counts()
#776 negative sentiments
# 1209 positive or neutral sentiments


sns.countplot(x='Year',data=stockdata,hue='sentiment')
# 2015 and 2016 had fairly high positive sentiments.It means it
#was a good year for stock market

sns.lineplot(x='Year',y='Close',data=stockdata)

sns.countplot(x='Month',data=stockdata,hue='sentiment')
# No seasonality found in terms of month

sns.lineplot(x='Month',y='Close',data=stockdata)


plt.figure(figsize=(14,8))
sns.countplot(x='Day',data=stockdata,hue='sentiment')
# No special seasonality except every 30th day of month is risky for stock market as sentiments are balanced for 0/1

sns.countplot(x='Dayofweek',data=stockdata,hue='sentiment')
sns.countplot(x='Dayofweek_name',data=stockdata,hue='sentiment')
# More postive news on friday

stockdata.groupby('sentiment')['Weekofyear'].value_counts()
# 39th week of yr has maximum negative sentiment. mostly september month-> In stock market terms its called SEPTEMBER EFFECT
# 2nd,49th week of year has maximum positive sentiment
# JANUARY Is considered a good month generally-> january effect

sns.countplot(x='Quarter',data=stockdata,hue='sentiment')
sns.countplot(x='semester',data=stockdata,hue='sentiment')

#--------------------------------------------------------
#visualization on closing price
sns.lineplot(x='Year',y='Close',data=stockdata)

#average of closing prices wrt each month
plt.plot(stockdata.groupby('Month')['Close'].mean())
# sept,oct are worst months as revealed by our sentiment above also

plt.plot(stockdata.groupby('Day')['Close'].mean())
#no definite trend

plt.plot(stockdata.groupby('Dayofweek_name')['Close'].mean())
# friday see low closing price since next days are weekend

sns.boxplot(x="Dayofweek_name", y="Close", data=stockdata,palette='rainbow')

plt.plot(stockdata.groupby('Weekofyear')['Close'].mean())
#september effect

plt.figure(figsize=(12,5))
sns.boxplot(x="Weekofyear", y="Close", data=stockdata,palette='rainbow')

#39th-42nd week i.e sept-oct is worst, year end is very good

plt.plot(stockdata.groupby('Quarter')['Close'].mean())
#oct,nov,dec is worst quarter, summer months lie in best quarter

sns.boxplot(x="Quarter", y="Close", data=stockdata,palette='rainbow')

#----------------------------------------------------------------------
#plot closing price of each year 
# each colour indicates year
plt.figure(figsize=(12,5))
stockdata[stockdata['Year']==2008]['Close'].plot(label="2008")
stockdata[stockdata['Year']==2009]['Close'].plot(label="2009")
stockdata[stockdata['Year']==2010]['Close'].plot(label="2010")
stockdata[stockdata['Year']==2011]['Close'].plot(label="2011")
stockdata[stockdata['Year']==2012]['Close'].plot(label="2012")
stockdata[stockdata['Year']==2013]['Close'].plot(label="2013")
stockdata[stockdata['Year']==2014]['Close'].plot(label="2014")
stockdata[stockdata['Year']==2015]['Close'].plot(label="2015")
stockdata[stockdata['Year']==2016]['Close'].plot(label="2016")
plt.legend()
plt.show()

# see columns with which closing price has max correlation with
stockdata.corr()['Close'].sort_values(ascending=False).head(10)
# corr with retail sales, gdp,cpi

# lets visualize 
sns.jointplot(x='RETAIL_SALES_INCLFOOD',y='Close',kind='reg',data=stockdata)
# good correlation as points are closer to straight line, top and right bars show distplot of individual variables

sns.jointplot(x='RETAIL_SALES_EXCLFOOD',y='Close',kind='reg',data=stockdata)

sns.jointplot(x='GDP',y='Close',kind='reg',data=stockdata)

sns.jointplot(x='CPI',y='Close',kind='reg',data=stockdata)

sns.jointplot(x='PPI',y='Close',kind='reg',data=stockdata)
#As ppi had 0.5 corr, we can see how mediocre is this plot


#--------------------------------------------------------
# Create new column called daily returns so that we can find  return or how much drop or rise in close price each day
#if you multiply by 100 you get %change everyday
daily_returns = stockdata['Close'].pct_change()
stockdata['daily_returns'] = daily_returns
stockdata.corr()['daily_returns'].sort_values(ascending=False)
# daily returns has 23% corr with close->great 
#daily returns has 53% corr with sentiment->very grat

# Daily returns show increase(+value) or decrease(-value) wrt previous day    

#sns.lineplot(x='Date', y='daily_returns',data=stockdata)

stockdata['daily_returns'].max() # Maximum return is 11% on a day
stockdata['daily_returns'].min() # Maximum return is -7.8% on a day
# There was a sharp increase and decrease during year 2009  owing to stock market crash. 

# Plot of daily returns
fig = plt.figure()
ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
stockdata['daily_returns'].plot.hist(bins = 100)
ax1.set_xlabel("Daily returns %")
ax1.set_ylabel("Percent")
ax1.set_title("DJI daily returns data")
plt.show()
# the returns are quite volatile and the stock  can move +/- 2.5% on any given day

print(daily_returns.describe())
##  The plot looks normally distributed with mean value around 0

# Cumulative returns tell us how much we earn or loss in a given timeperod.say you invest 1$ today. how much will you earn totally on that 1 yr after 3 yrs say. 
#Calculated by summing each row of daily returns

cum_daily_returns = (1 + daily_returns).cumprod()
stockdata['cum_daily_returns']= cum_daily_returns

fig = plt.figure()
sns.relplot(x='Year',y='cum_daily_returns',data=stockdata,kind='line')
# as shown if you invest 1$ in dji stocks in 2008, you will get 1.45$ in 2016

#plot of cumulative daily returns
fig = plt.figure()
ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
stockdata['cum_daily_returns'].plot.hist(bins = 100)
ax1.set_xlabel(" Cum Daily returns")
ax1.set_ylabel("Percent")
ax1.set_title("DJI Cumulative daily returns data")
plt.show()

stockdata['cum_daily_returns'].max()
stockdata['cum_daily_returns'].min()

# There has been days when the prices dropped by 56%(1->0.55) and there have been days when stocks rose upto 156% if we see from 2008

sns.lineplot(x='Date',y='cum_daily_returns',data=stockdata)
# as we see it faced 56%drop in 2009 and its 156% hike around 2015-16

#find correlation of close and stockdata
stockdata.corr()['cum_daily_returns'].sort_values(ascending=False)
# we have 100% corr with close which is intuitive. if we know  cum_daily_returns for a period say after 365 days,we can
# easily calculate closing price on 365th day by simple maths.  we cant use this column as 100% corr will mess with our model
# lets see if we can use yesterday_cum_daily_return

stockdata['yesterday_cumdr'] = stockdata['cum_daily_returns'].shift()
stockdata.corr()['yesterday_cumdr'].sort_values(ascending=False)



#----------------------------------------------------------------------
# Finding the volatality i.e change in variance over the period

# highly volatile means more risky. Here we use stddev as metric. we use rolling std dev which takes as parameter a window no of days.
# within these days it gives equal weight to study risk. 
#we choose 75 days 
min_periods = 75
# Calculate the volatility
vol = daily_returns.rolling(min_periods).std() * np.sqrt(min_periods) 
stockdata['vol']=vol

# Plot the volatility
#sns.lineplot(x='Date',y='vol',data=stockdata)
# As the plot shows the risk of DJI has gradually decreased over the years


sns.boxplot(x=stockdata['Open'])
sns.boxplot(x=stockdata['High'])
sns.boxplot(x=stockdata['Low'])
sns.boxplot(x=stockdata['Close'])

# Plot a histogram for all the columns of the dataframe. This shows the frequency of values in all the columns
import matplotlib.pyplot as plt
sns.set()
stockdata.hist(sharex = False, sharey = False, xlabelsize = 4, ylabelsize = 4, figsize=(15, 15))

#heatmap
plt.figure(figsize=(20,20))
plt.title('Pearson correlation of continuous features', y=1.05, size=12)
sns.heatmap(stockdata.corr(),cmap='coolwarm',linecolor='white',linewidths=1,vmax=1.0, square=True,  annot=True)
plt.show()

# sentiment has strong reln with daily_returns
# year has strong correlation with many variables
# closing price has very strong corr with maxm economic variables we made
# sentiment has corr with daily returns,and okayish
#corr with our economic variables

#visualization done

#---------------------------------------------------------------------
# performing regression model

#since reg model does not expect nan values

stockdata.isnull().sum()

# we need to fill vol and cum_dailyreturns by bfill
stockdata['daily_returns'].fillna(method='bfill',inplace=True)
stockdata['cum_daily_returns'].fillna(method='bfill',inplace=True)
stockdata['yesterday_cumdr'].fillna(method='bfill',inplace=True)
stockdata['vol'].fillna(method='bfill',inplace=True)
stockdata.isnull().sum()

# drop columns that are not necessary for reg model
cols
stockdata2 = stockdata.copy()
stockdata=stockdata2.copy()
stockdata.drop(['Date','Dayofweek_name','headlines','cum_month','cum_daily_returns'],axis=1,inplace=True)
stockdata.columns

# Trend Analysis
sns.lineplot(x='Year',y='Housepricemedian',data=stockdata)
sns.lineplot(x='Year',y='Close',data=stockdata)
# similarly done for other columns 

stockdata.drop([
        'Housepricemedian', 'FEDFUNDS'
       , 'LEI',
        'NEWHOUSEUNITS', 
         'daily_returns',
       'yesterday_cumdr', 'vol'],axis=1,inplace=True)

# bad columns dropped. 
stockdata.info()
desc = stockdata.describe().transpose()

#----------------------------------------------------------------------
# Applying RNN LSTM
#20%split
data_train = stockdata.iloc[:1588]
data_test = stockdata.iloc[1588:]

data_train.corr()['Close']

#shifting close to 1st column(insert and drop)
closetrain=data_train['Close']

#good way of shifting one column to beginning
data_train.drop(labels=['Close','Open','High','Low'],axis=1,inplace=True)
data_train.insert(0, 'Close',value=closetrain)
trainingdata=data_train

trainingdata.isnull().sum()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()   
trainingdata = scaler.fit_transform(trainingdata)
scale = scaler.scale_
# scale now shows the values that were divided for each column

# Logic is that we train for first 60 days, test on 61st day
# then train for next 60 days, test on next day
step_size = 60 # days that are used for the following prediction

x_train = []
y_train = []
for i in range(step_size,trainingdata.shape[0]):                
    x_train.append(trainingdata[i-step_size:i,1:]) # putting all columns after close
    y_train.append(trainingdata[i,0]) # putting close column
# we convert it into array as LSTM works with mroe than 2dimension
    
x_train = np.array(x_train)                   
y_train = np.array(y_train)
print(x_train.shape)                               
#1528 rows,60 columns, 58 dimensions
print(y_train.shape)   

# prepare test dataset
# in testing we need last 60 days data for each cell since  that is req of lstm. so we add tail of train set to test set  so that there are past 60 days data for 1st observ. in test set

past_60days = data_train.tail(step_size)
closetest=data_test['Close']
data_test.drop(labels=['Close','Open','High','Low'],axis=1,inplace=True)
data_test.insert(0, 'Close',value=closetest)
dfz = past_60days.append(data_test,ignore_index=True,sort=False)

#apply scaler to test set
data_test = scaler.transform(dfz)


x_test = []
y_test=[]

for i in range(step_size,data_test.shape[0]):
    x_test.append(data_test[i-step_size:i,1:])
    y_test.append(data_test[i,0])
    

# converting into arrays as lstmt require more than 2 dimension   
x_test = np.array(x_test)     
y_test = np.array(y_test)
print(x_test.shape)                               
#397 rows,60 columns, 58 dimensions
print(y_test.shape)   
                            

x_train.shape[1]

#LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(50,activation="relu",return_sequences=True, input_shape=(x_train.shape[1],28)))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(60,activation="relu",return_sequences=True))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(80,activation="relu",return_sequences=False))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(1))

lstm_model.summary()
#93,641 parameters

lstm_model.compile(optimizer="adam",loss="MSE")
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
lstm_model.fit(x_train,y_train,epochs=1,batch_size=32,
               validation_data=(x_test, y_test), verbose=1,
          callbacks=[early_stop])

#plotting the losses for train and validation set
losses = pd.DataFrame(lstm_model.history.history)
losses
losses.plot()
#model is not overfitting 

# predicting using lstm model
y_pred = lstm_model.predict(x_test)
scale = scaler.scale_
scale = scale[0]
scale
# we need to scale back both ypred and ytest to compare
y_pred=y_pred*scale


y_test = y_test*(scale)

# visualizing results
plt.figure(figsize=(14,5))
plt.plot(y_test,color='red')
plt.plot(y_pred,color='blue')
plt.legend()

# Our model performs bad in the end side data


from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score
mae = mean_absolute_error(y_test, y_pred)
print(mae) #691
stockdata['Close'].describe()
# mean close price is 13464 dollars
(691/13464)*100
# our model deviates only 5% which is pretty good
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(rmse) #825
explained_variance_score(y_test,y_pred)

#execute seperately below code

#comparing model against ideal trend line
plt.scatter(y_test,y_pred)

# Perfect predictions
plt.plot(y_test,y_test,'r')

#--------------------------------------------

#applying other regression models
xr= stockdata.drop(['Close','Low','High'
                    
                    ],axis=1)
yr=stockdata["Close"]
x1= xr.iloc[0:1787]  #90 train 
x2 = xr.iloc[1788:] #10% test
xcopy = x2.copy()
y1= yr.iloc[0:1787]  
y2 = yr.iloc[1788:] 
from sklearn.linear_model import LinearRegression
lr = LinearRegression(fit_intercept=True)
lr.fit(x1,y1)    
yp = lr.predict(x2)
from sklearn.metrics import r2_score
print(r2_score(y2,yp)) #93.3%

#MLR with single train test
import statsmodels.api as smf
x1=np.append(arr=np.ones((1787,1)).astype(int),values=x1,axis=1)
x1_df = pd.DataFrame(x1) # for reference
from statsmodels.regression.linear_model import OLS
SL = 0.05
x1_opt=x1[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]]

def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = OLS(y1, x).fit()
        maxVar = max(regressor_OLS.pvalues)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
x1_optimal = backwardElimination(x1_opt, SL)
x1_optimal_df = pd.DataFrame(x1_optimal)
#44 COLUMNS REDUCED TO 11..check them manually
xr.columns

cols3=[ 'sentiment', 'Open',  'yesterday_open','yesterday_high','yesterday_low','yesterday_close','Volume','GDP','RETAIL_SALES_EXCLFOOD','CSIndex','RETAIL_SALES_INCLFOOD']
# extracting these columns in test set

x2=pd.DataFrame(x2)
x2 = x2[cols3]
x2=np.append(arr=np.ones((196,1)).astype(int),values=x2,axis=1)
x2_df= pd.DataFrame(x2)

lr.fit(x1_optimal,y1)     
yp_optimal = lr.predict(x2)
print(r2_score(y2,yp_optimal))
#93.6%

plt.figure(figsize=(12,4))
plt.plot(y2,c='y',label='Actual close')
plt.plot(yp_optimal,c='r',label='Predicted close')
plt.xlabel("Testing instances")
plt.ylabel("Close price")
plt.legend()
plt.show()
#--------------------------------------------------

#Random forestregression(CART)
xr.columns
stockdata.corr()['Close'].sort_values(ascending=False)
# As we see dropping columns results in lowering of r2 score
xr= stockdata.drop(['Close','Low','High'
                    
                    ],axis=1)
yr=stockdata["Close"]

x1= xr.iloc[0:1787] 
x2 = xr.iloc[1788:] 
y1= yr.iloc[0:1787]  
y2 = yr.iloc[1788:] 

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 100, random_state = 0)
rf.fit(x1,y1)   
yp = rf.predict(x2)
print(r2_score(y2,yp))

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y2, yp)
print(mae) #57
stockdata['Close'].describe()
# mean close price is 13464 dollars
(57/13464)*100
# (0.42%deviation)

# get importance
importance = rf.feature_importances_
feature_importance = pd.DataFrame(importance,columns=['importance'])
feature_importance = feature_importance.sort_values(by=['importance'],ascending=False)
# high,low,IPI matters most
colname=[]
for i in feature_importance.index:
    colname.append(x1.columns[i])
feature_importance['colname']=colname

#------------------------------------------------------------------------

# Decision tree train test split
xr= stockdata.drop(['Close','Low','High'
                    
                    ],axis=1)
yr=stockdata["Close"]

x1= xr.iloc[0:1787]  #90
x2 = xr.iloc[1788:] #10%
y1= yr.iloc[0:1787]  
y2 = yr.iloc[1788:] 
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(random_state=0)
dt.fit(x1,y1)

# Predicting a new result
yp = dt.predict(x2)
print(r2_score(y2,yp))
rmse = np.sqrt(mean_squared_error(y2,yp))
print(rmse)
mae = mean_absolute_error(y2,yp)
print(mae) #90
stockdata['Close'].describe()
# mean close price is 13464 dollars
(90/13464)*100
# our model deviates only 0.6% which is pretty good
explained_variance_score(y2,yp)
# explaineds 96% variance
plt.plot(y2)
plt.plot(yp,c='r')
#-------------------------------------------------------------------
#svr
xr= stockdata.drop(['Close','Low','High'
                    
                    ],axis=1)
yr=stockdata["Close"]

x1= xr.iloc[0:1787]  #90
x2 = xr.iloc[1788:] #10%
y1= yr.iloc[0:1787]  
y2 = yr.iloc[1788:] 
from sklearn.svm import SVR
svr = SVR(kernel = 'rbf')
svr.fit(x1,y1)   
yp = svr.predict(x2)
print(r2_score(y2,yp))
mae = mean_absolute_error(y2,yp)
print(mae) #90
(4492/13464)*100
#33% deviation
plt.plot(y2)
plt.plot(yp,c='r')
#bad model
#----------------------------------------------------------------------
# Conclusion:
# We tried out EDA understanding behavior of each and every variable
# We implemented various visualization plots to understand trend and behavior of variables across time
#We created various economic and non economic variables that contributed towards good regression results
# We implemented various regression models . MLR with backward elimination performed best
