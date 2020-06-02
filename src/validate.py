#psaw
from psaw import PushshiftAPI

api = PushshiftAPI()

import datetime as dt

start_epoch=int(dt.datetime(2017, 1, 1).timestamp())
end_epoch=int(dt.datetime(2020, 1, 1).timestamp())

headlines_data = list(api.search_submissions(after=start_epoch,
                             before=end_epoch,
                            subreddit='usanews',
                            filter=[ 'created','title']))
headlines_data = pd.DataFrame([thing.d_ for thing in headlines_data])

#change structure of date
def get_date(created):
        return dt.datetime.fromtimestamp(created)


timestamp = headlines_data["created"].apply(get_date)
headlines_data= headlines_data.assign(timestamp = timestamp)
#drop unix timestamp dates

headlines_data.drop(['created','created_utc'],axis=1,inplace=True)

# extract only date part from timestamp
for row in range(len(headlines_data)):
    headlines_data['timestamp'].iloc[row]=headlines_data['timestamp'].iloc[row].date()


x = headlines_data.groupby(by='timestamp')
count = x.count()
count.sort_values(ascending=False,by='title')

# get only those rows where total news>25. we ill be testing on more than top 25
y = count[x.count()>=25]
y =y.dropna()
y = y.reset_index()

#getting titles and no of news of one date together
news =pd.merge(headlines_data,y,on='timestamp')

#logic for making final dictionary of date with news as columns(key-> date,val->news titles) & exporting each file to csv format
i=1
for date in headlines_data['timestamp'].unique():
    y = headlines_data[headlines_data['timestamp']==date]
    key_value = {}
    list_key = []
    list_val = []  
    
    list_val2=[]
    list_key.append(str(date))
    
    list_key = str(list_key)
    
    for index, row in y.iterrows():
        list_val.append(row['title'])
    lenList = len(list_val)
        
    for elements in range(0,lenList) :
        
        list_val2.append(list_val[elements])
    key_value[list_key] = list_val2
    data = pd.DataFrame(key_value)
    data=data.transpose()
    #file_name_str = str(data.columns[0]) i
    i = str(i)
    file_name = i +".csv"
    data.to_csv(file_name)
    i=int(i)+1
#csv files exported
    
#---------------------------------------------------------------------    
# combining all csv files in the directory
import glob
import pandas as pd

extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])

#export to csv
combined_csv.to_csv( "combined.csv", index=False, encoding='utf-8-sig')

#make changes to combined_csv (insert column names at top row and make it as combinednew.csv)
#--------------------------------------------------------------------


livenews= pd.read_csv('combinednew.csv',error_bad_lines=False)
from datetime import datetime

#changing date to right format
date_object=[]
x=livenews['date']
for row in x:
    list=[]
    for i in range(2,12): 
        list.append(row[i])
    newstr = ''.join(list)
    date_object.append( datetime.strptime(newstr, '%Y-%m-%d').date())
date_object=pd.DataFrame(date_object)  
livenews['date']=date_object
  
livenews['date'].nunique()
livenews.notnull().any(axis = 0)

# There is no column where all values are null. Thus some dates contain
#30, while some contain 70 news for that date

list=['Date']
for i in range(0,73):
    list.append(str(i))
livenews.columns=list

# we will apply nlp on above test set to find sentiment and get news only first
livenews2=livenews.drop('Date',axis=1)
livenews2 = livenews2.replace('b\"|b\'|\\\\|\\\"', '', regex=True)
livenews2.replace("[^a-zA-Z]"," ",regex=True, inplace=True)
livenews2=livenews2.replace(np.nan," ")
livenews=livenews.replace(np.nan," ")

paras = []
for row in range(0,len(livenews2.index)):
    paras.append(' '.join(str(x) for x in livenews2.iloc[row,0:73]))
    
    
paras[0]    # news for 31st dec2019
bowparas=countvector.transform(paras)
from sklearn.feature_extraction.text import TfidfTransformer
bowparas = tfidf.transform(bowparas)
#predicting sentiments with our best classifier
sentiments = model_xgb.predict(bowparas)
livenews['sentiment']=sentiments

livestock = pd.read_csv('livestockdata.csv')

date_object=[]
x=livestock['Date']
for row in x:
    list=[]
    for i in range(0,10): 
        list.append(row[i])
    newstr = ''.join(list)
    date_object.append( datetime.strptime(newstr, '%Y-%m-%d').date())
date_object=pd.DataFrame(date_object)  
livestock['Date']=date_object

livenews = livenews.sort_values(by='Date')

stock =pd.merge(livestock,livenews,on='Date')
list=['Date', 'Open', 'High', 
             'Low', 'Close', 'Adj Close', 'Volume'
             ,'sentiment']
# assigning columns
stock=stock[list]

#convert datetime.date to datetime.datetime
list=[]
for i in stock['Date']:
    list.append(datetime.combine(i, datetime.min.time()))
stock['Date']=list
stock['Date']= pd.to_datetime(stock['Date'])

# modifying columns to stock
stock.drop(['Adj Close'],axis=1,inplace=True)
stock['Volume'] = stock['Volume'].astype(float)
type(stock['Date'].iloc[0])

stock['Year'] = stock['Date'].dt.year
stock['Month'] = stock['Date'].dt.month
stock['Day'] = stock['Date'].dt.day
stock['Quarter'] = stock['Date'].dt.quarter
stock['semester'] = np.where(stock['Quarter'].isin([1,2]),1,2)
stock['Dayofweek'] = stock['Date'].dt.dayofweek
stock['Dayofweek_name'] = stock['Date'].dt.day_name
stock['Dayofyear'] = stock['Date'].dt.dayofyear
stock['Weekofyear'] = stock['Date'].dt.weekofyear
stock['yesterday_open'] = stock['Open'].shift()
stock['yesterday_close'] = stock['Close'].shift()
stock['yesterday_high'] = stock['High'].shift()
stock['yesterday_low'] = stock['Low'].shift()
stock = stock.fillna(method='bfill')
cols = ['Date', 'Year', 'Month','Day', 'Dayofweek','Dayofweek_name'
        ,'Dayofyear','Weekofyear',
       'Quarter', 'semester', 'sentiment',
       'Open','yesterday_open', 'High','yesterday_high',
       'Low','yesterday_low', 'Close','yesterday_close',
       'Volume'
       ]
#Rearranging columns for simplicity
stock = stock.reindex(columns=cols)
stock2=stock
#--------------------- 
# getting economic variables
start='2017/04/06'
end='2019/12/30'
dataset=stock2
qstart='2017/04/01'
qend='2020/01/01'
mstart='2017/04/01'
mend='2020/01/01'
ystart='2017/01/01'
yend='2020/01/01'
wstart='2017/03/10'
wend='2020/01/10'

stock2= economic(start,end,dataset)
#economic variables made
#--------------------------------------------------------------------
daily_returns = stock2['Close'].pct_change()
stock2['daily_returns'] = daily_returns

cum_daily_returns = (1 + daily_returns).cumprod()
stock2['cum_daily_returns']= cum_daily_returns
stock2['yesterday_cumdr'] = stock2['cum_daily_returns'].shift()

min_periods = 75
vol = daily_returns.rolling(min_periods).std() * np.sqrt(min_periods) 
stock2['vol']=vol

stock2['daily_returns'].fillna(method='bfill',inplace=True)
stock2['cum_daily_returns'].fillna(method='bfill',inplace=True)
stock2['yesterday_cumdr'].fillna(method='bfill',inplace=True)
stock2['vol'].fillna(method='bfill',inplace=True)
stock2.isnull().sum()

stock3 = stock2.copy()
stock2=stock3.copy()
#stock2=stock3
stock2.drop(['Date','Dayofweek_name','cum_daily_returns'],axis=1,inplace=True)
stock2.drop([
        'Housepricemedian', 'FEDFUNDS'
       , 'LEI',
        'NEWHOUSEUNITS', 
         'daily_returns',
       'yesterday_cumdr', 'vol'],axis=1,inplace=True)

#---------------------------------------------------------------------
# applying our previously tained reg model

xr2_lr= stock2.drop(['Close','Low','High'],axis=1)
cols3=[ 'sentiment', 'Open',  'yesterday_open','yesterday_high','yesterday_low','yesterday_close','Volume','GDP','RETAIL_SALES_EXCLFOOD','CSIndex','RETAIL_SALES_INCLFOOD']
xr2_lr = xr2_lr[cols3]
xr2_lr=np.append(arr=np.ones((687,1)).astype(int),values=xr2_lr,axis=1)
xr2_lr= pd.DataFrame(xr2_lr)
yr2=stock2["Close"]


from sklearn.linear_model import LinearRegression
yp2lr = lr.predict(xr2_lr)
from sklearn.metrics import r2_score
print(r2_score(yr2,yp2lr)) #98%

plt.figure(figsize=(12,4))
plt.plot(yr2,c='b',label='Actual close')
plt.plot(yp2lr,c='r',label='Predicted close')
plt.xlabel("Testing instances")
plt.ylabel("Close price")
plt.legend()
plt.show()

#--------------------------------------------------------------------
#random forest
xr2= stock2.drop(['Close','Low','High'],axis=1)
yr2=stock2["Close"]
yp2 = rf.predict(xr2)
print(r2_score(yr2,yp2))
#----------------------------------------------------------------------
#decision tree
xr2= stock2.drop(['Close','Low','High'],axis=1)
yr2=stock2["Close"]

yp2 = dt.predict(xr2)
print(r2_score(yr2,yp2))

#------------------------------------------------------------------
#svr
xr2= stock2.drop(['Close','Low','High'],axis=1)
yr2=stock2["Close"]
yp2 = svr.predict(xr2)
print(r2_score(yr2,yp2))
#------------------------------------------------------
# direcn
# reexecute MLR

# predicting movement
yest_close = stock2['yesterday_close']
date=stock3['Date']
date = pd.DataFrame(date)
yp2lr=pd.DataFrame(yp2lr)
yp2lr= pd.concat([yp2lr,date],axis=1)
yp2lr['yesterday_close']=yest_close

yp2lr.columns=['predictions','Date','yesterday_close']

yr2 = pd.DataFrame(yr2)
yr2= pd.concat([yr2,date],axis=1)
yr2['yesterday_close']=yest_close


rand = np.random.randint(low=0,high=687,size=1)
d = yp2lr.iloc[rand]
pred= d['predictions'].values
yest_act=d['yesterday_close'].values
datetoday= d['Date'].values.astype(str)

yest_act

# str to date
date_object1=[]
for row in datetoday:
    list=[]
    for i in range(0,10): 
        list.append(row[i])
    newstr1 = ''.join(list)
    date_object1.append( datetime.strptime(newstr1, '%Y-%m-%d').date())

print("\n \nToday's date is", date_object1[0])
if pred>yest_act:
    print("Closing price yesterday was {}".format(yest_act))
    print("Stock price today should close at{}, +{} from yesterday's price".format( pred,(pred-yest_act)))
   
else:
    print(" Closing price yesterday was {}".format(yest_act))
    print("Stock price today should close at {}, -{} from yesterday's price".format(pred,(yest_act-pred)))


    
    