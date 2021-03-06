import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxopt as opt
from cvxopt import blas, solvers


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
sprice_all_df = pd.DataFrame(sfile)
sprice_1999to2014_df = sprice_all_df.iloc[0:193]
sprice_2015to2019_df = sprice_all_df.iloc[193:]


# Reading Covariance
cov = pd.read_excel(filename, sheet_name="Covariance")
cov = cov.iloc[1:, 2:]
covmatrix = np.array(cov)
print("covariance")
print(cov)


# Reading Returns
ret_all = {}
col = "Return"

for sheet in sheetnames:
	terms = []
	df_terms = pd.read_excel(filename, sheet_name = sheet)
	for index, row in df_terms.iterrows():
		if index == 0:
			terms.append(0)
		else:
			terms.append(row[col])
	ret_all[sheet] = terms

ret_all_df = pd.DataFrame(ret_all)
ret_1999to2014_df = ret_all_df.iloc[0:193]
ret_2015to2019_df = ret_all_df.iloc[193:]

# Get average return
avgreturn = []
for i, column in enumerate(ret_all_df):
	avgreturn.append(ret_all_df[column].mean())
avgreturn = np.array(avgreturn)
print("avg returns for each stock")
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
plt.xlabel('Volatility (Std. Deviation)')
plt.ylabel('Expected Returns')
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

# (1) create portfolio(x) w/ optimal weight

#----- Optimization ---------
def get_optimal_weight(returns):
	solvers.options['show_progress'] = False
	n = len(returns)
	# returns = np.asmatrix(returns)

	N = 100
	mus = [10**(5.0 * t/N - 1.0) for t in range(N)]

	# cvopt objective
	S = opt.matrix(np.cov(returns))
	pbar = opt.matrix(np.mean(returns, axis=1))
	# constraints
	G = -opt.matrix(np.eye(n))   
	h = opt.matrix(0.0, (n ,1))
	A = opt.matrix(1.0, (1, n))
	b = opt.matrix(1.0)

	portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] 
				  for mu in mus]
	returns = [blas.dot(pbar, x) for x in portfolios]
	risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]

	m1 = np.polyfit(returns, risks, 2)
	x1 = np.sqrt(m1[2] / m1[0])
	wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
	return wt

# assign solved optimal weight
wt = get_optimal_weight(ret_1999to2014_df.T)
x_weights = np.array(wt).flatten()
# print("solve optimal weights")
# print(x_weights)

x_monthly_expected_return = []
for i, row in ret_2015to2019_df.iterrows():
	x_monthly_return = row
	x_monthly_expected_return.append(np.dot(x_weights, x_monthly_return))


# (2) create portfolio(y) w/ equal weight
y_weight = np.full(21, 1/21)
y_monthly_expected_return = []
for i, row in ret_2015to2019_df.iterrows():
	y_monthly_return = row
	y_monthly_expected_return.append(np.dot(y_weight, y_monthly_return))

#========
# Part A
#========
print("\n============== Problem 2(a) ==================\n")

x_monthly_expected_return = np.array(x_monthly_expected_return)
x_mean = np.mean(x_monthly_expected_return)
x_sigma = np.std(x_monthly_expected_return)
print("x mean, x std: ")
print(x_mean, x_sigma)

y_monthly_expected_return = np.array(y_monthly_expected_return)
y_mean = np.mean(y_monthly_expected_return)
y_sigma = np.std(y_monthly_expected_return)
print("y mean, y std: ")
print(y_mean, y_sigma)

#========
# Part B
#========

# Answer:
# The results we got are
# x mean: 1.0397481140644071
# x_std: 0.1353880658257495
# y mean: 1.0091138793483676
# y std: 0.04302419027382689

# Portfolio X with optimal weights performs better.
# However, the risk associated with it is also higher



#***************************************************
# Problem 3
#***************************************************

#========
# Part A
#========
print("\n============== Problem 3(a) ==================\n")

# Compute M Scores
def compute_M_scores(n, m, lastidx = 193):
	# print("December 2014 stock price")
	
	prices = sprice_1999to2014_df.iloc[lastidx-1:lastidx].values[0] # Dec2014
	prices_t_1 = sprice_1999to2014_df.iloc[lastidx-1-1:lastidx-1].values[0]
	prices_t_n = sprice_1999to2014_df.iloc[lastidx-1-n:lastidx-n].values[0]
	prices_t_m = sprice_1999to2014_df.iloc[lastidx-1-m:lastidx-m].values[0]
	# print(sprice_1999to2014_df)
	# print(prices_t_1)
	# print(prices_t_n)
	# print(prices_t_m)

	mscores = []
	for i in range(0, len(prices)):
		R1 = np.log(prices_t_1[i]/prices_t_n[i])
		R2 = np.log(prices_t_m[i]/prices_t_1[i])
		M = 0.5*R1 + 0.5*R2
		mscores.append(M)

	return mscores


x_meanvar = []
y_meanvar = []
for n in [1, 2, 3, 4, 5, 6]:
	for m in [3, 4, 5, 6, 7, 8, 9]:
		# compute M scores
		mscores = compute_M_scores(n=n, m=m)
		# print(mscores)

		# pick top mscores and corresponding column idx
		mscores = np.array(mscores)
		maxidx = mscores.argsort()[-10:][::-1]
		maxidx = np.sort(maxidx)
		# print(maxidx)

		# get monthly return of top 10 stocks from 1999 to 2014
		ret_1999to2014_df_top10 = ret_1999to2014_df.iloc[:, maxidx]
		ret_2015to2019_df_top10 = ret_2015to2019_df.iloc[:, maxidx]


		# ----Then repeat Problem 2-------
		# (1) create portfolio(x) w/ optimal weight
		wt = get_optimal_weight(ret_1999to2014_df_top10.T)
		x_weights = np.array(wt).flatten()
		# print("solve optimal weights")
		# print(x_weights)
		x_monthly_expected_return = []
		for i, row in ret_2015to2019_df_top10.iterrows():
			x_monthly_return = row
			x_monthly_expected_return.append(np.dot(x_weights, x_monthly_return))

		# (2) create portfolio(y) w/ equal weight
		y_weight = np.full(10, 1/10)
		y_monthly_expected_return = []
		for i, row in ret_2015to2019_df_top10.iterrows():
			y_monthly_return = row
			y_monthly_expected_return.append(np.dot(y_weight, y_monthly_return))

		x_monthly_expected_return = np.array(x_monthly_expected_return)
		x_mean = np.mean(x_monthly_expected_return)
		x_sigma = np.std(x_monthly_expected_return)
		# print("x mean, x std: ")
		# print(x_mean, x_sigma)

		y_monthly_expected_return = np.array(y_monthly_expected_return)
		y_mean = np.mean(y_monthly_expected_return)
		y_sigma = np.std(y_monthly_expected_return)
		# print("y mean, y std: ")
		# print(y_mean, y_sigma)

		# Append results
		x_meanvar.append([x_mean, x_sigma, n, m])
		y_meanvar.append([y_mean, y_sigma, n, m])

print("\nportfolio for different n, m")
x_meanvar = pd.DataFrame(x_meanvar)
y_meanvar = pd.DataFrame(y_meanvar)
print("\nportfolio x: mean, std, n, m")
print(x_meanvar)
print("\nportfolio y: mean, std, n, m")
print(y_meanvar)


#========
# Part B
#========
# The best performing (regards to expected return) portfolio was acheived with n=6, m=6,
# where the expected return was 1.029630 and the volatility was 0.120398.

# The corresponding performance of the equal weight portfolio achieved an 
# expected return of 1.010566  and the volatility of 0.060045.


# The expected return for the mean variance efficient portfolio x_s has 3 times more expected return 
# with only 2 times more volatility than the of equal weighting portfolio y_s. 
# Therefore, we can conclude that the mean vairance efficient portfolio x_s performs better overall.

