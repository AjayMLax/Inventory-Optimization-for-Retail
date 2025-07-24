import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from xgboost import XGBRegressor
from gurobipy import Model, GRB, quicksum

## Import and explore data for patterns

dfd = pd.read_csv(".spyder-py3/OR projects/Inventory Optimization for retail/Data/demand_forecasting.csv", parse_dates=["Date"])

# Grouping by weekly sales to have better granularity of data

dfd['Week'] = dfd['Date'].dt.to_period('W').apply(lambda r: r.start_time)
dfd['Week'] = pd.to_datetime(dfd['Week'])

# Convert 'None' strings percieved as empty into 'None' string

for col in ['Seasonality Factors', 'External Factors']:
    dfd[col] = dfd[col].fillna('None')
dfd_full = dfd.copy()

# Looking at some examples to check if there are patterns

df_group = dfd_full.groupby(['Week', 'Product ID', 'Store ID'])['Sales Quantity'].sum().reset_index()
example_products = [7860, 4555, 6168]
for ep in example_products:
    df_p = df_group[(df_group['Product ID'] == ep)]
    plt.figure(figsize=(10,5))
    plt.plot(df_p['Week'], df_p['Sales Quantity'])
    plt.title(f"Sales Quantity over time for product ID {ep}")
    plt.xlabel("Week")
    plt.ylabel("Sales Quantity")
    plt.xticks(rotation=45)  
    plt.tight_layout()       
    plt.grid(True)
    plt.show()

## Since there are no patterns, try XGBoost training

# Convert columns with strings into numerical values using dictionaries and one-hot encoding

dfd['Promotions'] = dfd['Promotions'].map({'Yes' : 1, 'No' : 0})
dfd['Demand Trend'] = dfd['Demand Trend'].map({'Increasing' : 1, 'Stable' : 0, 'Decreasing' : -1})
dfd = pd.get_dummies(dfd, columns=['Seasonality Factors', 'Customer Segments', 'External Factors'])

features = ['Promotions', 'Seasonality Factors_Festival', 'Seasonality Factors_Holiday', 'Seasonality Factors_None', 'Customer Segments_Budget', 'Customer Segments_Premium', 'Customer Segments_Regular', 'External Factors_Competitor Pricing', 'External Factors_Economic Indicator', 'External Factors_None', 'External Factors_Weather']
df_group_encoded = dfd.groupby(['Week', 'Product ID'])[features + ['Sales Quantity']].mean().reset_index()

X = df_group_encoded[features]
y = df_group_encoded['Sales Quantity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

modelXG = XGBRegressor(random_state=42)
modelXG.fit(X_train, y_train)
y_predXG = modelXG.predict(X_test)

print("\nErrors of XGBoost model:")
rmseXG = sqrt(mean_squared_error(y_test, y_predXG))
r2XG = r2_score(y_test, y_predXG)
print(f"Root Mean Squared Error : {rmseXG:.2f}")
print(f"R² Score : {r2XG:.4f}")


## Forecast for the next 4 weeks

# Since there is no clear pattern and R² scores are almost 0, average weekly demand will be used to forecast

prod_avg = df_group.groupby('Product ID')['Sales Quantity'].mean()
last_week = df_group['Week'].max()
forecast_weeks = [last_week + pd.Timedelta(weeks = i) for i in range(1,5)]

df_forecast = pd.merge(pd.DataFrame({'Product ID': prod_avg.index}), pd.DataFrame({'Week': forecast_weeks}), how='cross')
df_forecast['Sales Forecast'] = df_forecast['Product ID'].map(prod_avg)


## Begin Modeling and Optimization

dfi = pd.read_csv(".spyder-py3/OR projects/Inventory Optimization for retail/Data/inventory_monitoring.csv")
dfp = pd.read_csv(".spyder-py3/OR projects/Inventory Optimization for retail/Data/pricing_optimization.csv")

df_di = pd.merge(df_forecast, dfi, on= 'Product ID', how='left')
df_dpi = pd.merge(df_di, dfp, on='Product ID', how='left')

# Chosen Modelling variables : Sales Forecast, Price, Storage Cost, Stock Levels, Warehouse Capacity, Reorder Point
# Dropping rows with empty values in modeling variable columns and changing return rate into decimals

df_opt = df_dpi.dropna(subset=['Sales Forecast', 'Price', 'Storage Cost', 'Stock Levels', 'Warehouse Capacity', 'Reorder Point'])
df_opt = df_opt.copy()
df_opt['Return Rate (%)'] = df_opt['Return Rate (%)']/100

# Convert columns into dictionaries with Product ID as the key

price = df_opt.set_index('Product ID')['Price'].to_dict()
storage = df_opt.set_index('Product ID')['Storage Cost'].to_dict()
stock = df_opt.set_index('Product ID')['Stock Levels'].to_dict()
capacity = df_opt.set_index('Product ID')['Warehouse Capacity'].to_dict()
reorder = df_opt.set_index('Product ID')['Reorder Point'].to_dict()
return_rate = df_opt.set_index('Product ID')['Return Rate (%)'].to_dict()
stockout_freq = df_opt.set_index('Product ID')['Stockout Frequency'].to_dict()
lead_time = df_opt.set_index('Product ID')['Supplier Lead Time (days)'].to_dict()
order_fulfill = df_opt.set_index('Product ID')['Order Fulfillment Time (days)'].to_dict()

# Create dictionary of sales forecast and include only corresponding produc-week pairs

sales = df_opt.set_index(['Product ID', 'Week'])['Sales Forecast'].to_dict()
prod_week = list(sales.keys())
prod = sorted({p for p,_ in prod_week})
week = sorted({w for _,w in prod_week})
first_week = week[0]

# Adjust reorder amount with a small buffer using Stockout frequency with alpha

alpha = 0.05 # Parameter that determines how much the stockout frequency affects the reorder point
reorder = {p : reorder[p]*(1 + alpha*stockout_freq[p]) for p in prod}

# Compute effective delay in number of weeks, rouding up

total_delay = {p: max(1, int(np.ceil((lead_time[p] + order_fulfill[p])/7))) for p in prod}

# Create Model and variables

m = Model('inventory')
order = m.addVars(prod_week, lb = 0, name='order')
inv = m.addVars(prod_week, lb = 0, name='inv')

# Add constraints

m.addConstrs((inv[p,first_week] == stock[p] + order[p,first_week] - sales[p,first_week] for (p,w) in prod_week if w == first_week), name = 'initial inv')
m.addConstrs((inv[p,w] <= capacity[p] for (p,w) in prod_week), name='inv_ub')
m.addConstrs((inv[p,w] >= reorder[p] for (p,w) in prod_week), name='inv_lb')
for i in range(1, len(week)):
    wk = week[i]
    
    for p in prod:
        wk_prev = week[i-1]
        delay = total_delay[p]
        
        if i-delay >= 0:
            wk_delay = week[i-delay]
            m.addConstr((inv[p,wk] == inv[p,wk_prev] + order[p,wk_delay] - sales[p,wk]), name = 'update inv')

# Add objective and optimize the model

obj = quicksum(storage[p]*inv[p,w] + return_rate[p]*price[p]*order[p,w] for (p,w) in prod_week)
m.setObjective(obj, GRB.MINIMIZE)
m.optimize()

## Save results in csv file

results = []
for (p,w) in prod_week:
    results.append({'Product ID' : p, 'Week' : w, 'Orders' : order[p,w].X, 'Inventory' : inv[p,w].X})

df_results = pd.DataFrame(results)
df_results.to_csv(".spyder-py3/OR projects/Inventory Optimization for retail/Results/Continuous_results.csv", index=False)
print("\nContinuous results exported to `Continuous_results.csv`")

df_rounded = df_results.copy()
df_rounded['Orders'] = np.ceil(df_rounded['Orders']).astype(int)
df_rounded['Inventory'] = np.ceil(df_rounded['Inventory']).astype(int)
df_rounded.to_csv(".spyder-py3/OR projects/Inventory Optimization for retail/Results/Rounded_results.csv", index=False)
print("\nRounded results exported to `Rounded_results.csv`")

## Plot results for a seclection of product IDs

sample_products = [1000, 1001, 1003] 

df_plot = pd.merge(df_rounded, df_opt[['Product ID', 'Week', 'Sales Forecast']], on=['Product ID', 'Week'])

plt.figure(figsize=(12, 6))
colours = ['blue', 'orange', 'green']
for i, pid in enumerate(sample_products):
    subset = df_plot[df_plot['Product ID'] == pid]
    colour = colours[i]
    
    plt.plot(subset['Week'], subset['Inventory'], marker='o', color = colour, linestyle='-', label=f'Inventory - {pid}')
    plt.plot(subset['Week'], subset['Orders'], marker='s', color = colour, linestyle='--', label=f'Orders - {pid}')
    plt.plot(subset['Week'], subset['Sales Forecast'], marker='x', color = colour, linestyle=':', label=f'Sales - {pid}')

plt.title('Inventory, Orders, and Sales Forecast over Weeks (Rounded)')
plt.xlabel('Week')
plt.ylabel('Quantity')
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(True)
plt.show()
