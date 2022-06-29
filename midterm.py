"""
python3.7
numpy
pyplot
pip install xlrd==1.2.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#***************************************************
# Reading Data
#***************************************************
filename = "ISYE6227_project_dataset.xlsx"

df = pd.read_excel(filename, sheet_name = None)
# allsheetnames = list(df.keys())
# sheetnames = allsheetnames[:-21] #-3
sheetnames = ['CVS','KO','PG','DWDP','GE','COP','CVX','MSFT','CSCO','CAH','MCK','JPM','BAC','VLO','TGT','HD','ADM','BA','F','VZ','KR']
print(sheetnames)

col = "Adj Close"
sfile = {}
for sheet in sheetnames:
	terms = []
	df_terms = pd.read_excel(filename, sheet_name = sheet)
	for index, row in df_terms.iterrows():
		terms.append(row[col])
	sfile[sheet] = terms

# Reading Covariance
cov = pd.read_excel(filename, sheet_name="Covariance")
cov = cov.iloc[1:, 2:]
covmatrix = np.array(cov)
print(cov)


# Reading Returns
avgreturn = []
retall = []
ret = pd.read_excel(filename, sheet_name="Condensed r and R")
ret = ret.iloc[1:,2:]
for i, column in enumerate(ret):
	if i%2 == 0:
	    retall.append(ret[column])
	    avgreturn.append(ret[column].mean())
avgreturn = np.array(avgreturn)
print(avgreturn)


#***************************************************
# Problem 1
#***************************************************

#========
# Part A
#========
meanstd_pair = []

for stock in sfile:
	mean = sum(sfile[stock])/len(sfile[stock])
	var = sum([((x - mean) ** 2) for x in sfile[stock]])/len(sfile[stock])
	std = var ** 0.5
	meanstd_pair.append([mean, std])
	# print("stock, "+str(mean)+" "+str(std))
# print(meanstd_pair)

meanstd_pair = np.array(meanstd_pair)
plt.figure()
plt.scatter(meanstd_pair[:,0], meanstd_pair[:,1])
plt.title('1(a) r-sigma diagram')
plt.show()


#========
# Part B
#========
# empty lists to store returns, volatility and weights of imiginary portfolios
port_returns = []
port_volatility = []
stock_weights = []
sharpe_ratio = []

# set the number of combinations for random portfolios
stk_set = sheetnames  
num_assets = len(stk_set)
num_portfolios = 500

#set random seed for reproduction's sake
np.random.seed(101)

# populate the empty lists with each portfolios returns,risk and weights
for single_portfolio in range(num_portfolios):
    # weights = np.random.random(num_assets)
    weights = np.random.uniform(low=-1, high=1, size=num_assets)
    weights /= np.sum(weights)
    returns = np.dot(weights, avgreturn)
    volatility = np.sqrt(np.dot(weights.T, np.dot(covmatrix, weights)))

    port_returns.append(returns)
    port_volatility.append(volatility)
    stock_weights.append(weights)

# a dictionary for Returns and Risk values of each portfolio
portfolio = {'Returns': port_returns,
             'Volatility': port_volatility}

# # extend original dictionary to accomodate each ticker and weight in the portfolio
for counter,symbol in enumerate(stk_set):
    portfolio[symbol+' weight'] = [weight[counter] for weight in stock_weights]

# # make a nice dataframe of the extended dictionary
df = pd.DataFrame(portfolio)

# # get better labels for desired arrangement of columns
column_order = ['Returns', 'Volatility'] + [stock+' weight' for stock in stk_set]

# # reorder dataframe columns
df = df[column_order]
df.head()

### Heat map plot of portfolios
plt.style.use('seaborn-dark')
df.plot.scatter(x='Volatility', y='Returns', cmap='RdYlGn', edgecolors='black', figsize=(10, 8), grid=True)

plt.xlabel('Volatility (Std. Deviation)')
plt.ylabel('Expected Returns')
plt.title('1(b) Efficient Frontier')
plt.show()


#========
# Part C
#========
# empty lists to store returns, volatility and weights of imiginary portfolios
port_returns = []
port_volatility = []
stock_weights = []
sharpe_ratio = []

# set the number of combinations for random portfolios
stk_set = sheetnames  
num_assets = len(stk_set)
num_portfolios = 500

#set random seed for reproduction's sake
np.random.seed(101)

# populate the empty lists with each portfolios returns,risk and weights
for single_portfolio in range(num_portfolios):
    # weights = np.random.random(num_assets)
    weights = np.random.uniform(low=0, high=1, size=num_assets)
    weights /= np.sum(weights)
    returns = np.dot(weights, avgreturn)
    volatility = np.sqrt(np.dot(weights.T, np.dot(covmatrix, weights)))

    port_returns.append(returns)
    port_volatility.append(volatility)
    stock_weights.append(weights)

# a dictionary for Returns and Risk values of each portfolio
portfolio = {'Returns': port_returns,
             'Volatility': port_volatility}

# # extend original dictionary to accomodate each ticker and weight in the portfolio
for counter,symbol in enumerate(stk_set):
    portfolio[symbol+' weight'] = [weight[counter] for weight in stock_weights]

# # make a nice dataframe of the extended dictionary
df = pd.DataFrame(portfolio)

# # get better labels for desired arrangement of columns
column_order = ['Returns', 'Volatility'] + [stock+' weight' for stock in stk_set]

# # reorder dataframe columns
df = df[column_order]
df.head()

### Heat map plot of portfolios
plt.style.use('seaborn-dark')
df.plot.scatter(x='Volatility', y='Returns', cmap='RdYlGn', edgecolors='black', figsize=(10, 8), grid=True)

plt.xlabel('Volatility (Std. Deviation)')
plt.ylabel('Expected Returns')
plt.title('1(c) Efficient Frontier')
plt.show()



#***************************************************
# Problem 2
#***************************************************

w_equal = np.full