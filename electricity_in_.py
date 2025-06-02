import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np
from xgboost import XGBRegressor
import seaborn as sns
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity

def create_weekly_agg(df, value_cols, date_col='settlement_date'):
    week_info = df[date_col].dt.isocalendar()
    df = df.copy()
    df['year'] = week_info['year']
    df['week'] = week_info['week'].astype('int32')
    weekly = df.groupby(['year', 'week'])[value_cols].sum().reset_index()
    weekly['year_week'] = weekly['year'].astype(str) + '-W' + weekly['week'].astype(str).str.zfill(2)
    return weekly

def mse_cosine_obj(y_pred, dtrain):
    y_true = dtrain.get_label()
    grad_mse = y_pred - y_true  
    
    norm_true = np.linalg.norm(y_true) + 1e-8
    norm_pred = np.linalg.norm(y_pred) + 1e-8
    dot = np.dot(y_true, y_pred)
    lambda_cos = 1000.0  
    
   
    grad_cos = lambda_cos * (
        y_true / (norm_true * norm_pred) - 
        (dot * y_pred) / (norm_true * norm_pred**3)
    )
    
    grad_total = grad_mse + grad_cos
    hess = np.ones_like(y_true)
    return grad_total, hess


data = pd.read_csv('historic_demand_2009_2024_noNaN.csv')
data['settlement_date'] = pd.to_datetime(data['settlement_date'])
print(data.describe(),data.info(),data['england_wales_demand'].describe())
data['year'] = data['settlement_date'].dt.year
data['month'] = data['settlement_date'].dt.month.astype('int32')
data['week'] = data['settlement_date'].dt.isocalendar().week.astype('int32')

max_date = data['settlement_date'].max()
start_date = max_date - pd.DateOffset(months=6)
filtered = data[(data['settlement_date'] > start_date) & (data['settlement_date'] <= max_date)]

weekly = create_weekly_agg(
    filtered,
    ['embedded_wind_generation', 'embedded_solar_generation', 'england_wales_demand', 'nd', 'tsd']
)

plt.figure(figsize=(12, 6))
for col in weekly.columns[2:-1]:
    plt.plot(weekly['year_week'], weekly[col], label=col, marker='o')
plt.title('Energy consumption by week (last 6 months)')
plt.xlabel('Year-Week')
plt.ylabel('Value')
plt.xticks(rotation=90, fontsize=8)
plt.grid(True)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

numeric_data = data.select_dtypes(include='number')
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation matrix')
plt.tight_layout()
plt.show()

features = [
    'embedded_wind_generation',
    'embedded_solar_generation',
    'embedded_wind_capacity',
    'embedded_solar_capacity',
    'non_bm_stor',
    'tsd',
    'year',
    'month',
    'week'
]

X = data[features]
y = data['england_wales_demand']

split_date = data['settlement_date'].quantile(0.8)
train = data[data['settlement_date'] <= split_date]
test = data[data['settlement_date'] > split_date]
X_train, y_train = train[features], train['england_wales_demand']
X_test, y_test = test[features], test['england_wales_demand']

param_grid = {
    'n_estimators': [200,],
    'max_depth': [5],
    'learning_rate': [0.05],
    'subsample': [1.0],
    'colsample_bytree': [0.8]
}

model = XGBRegressor(random_state=42)
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=3,
    n_jobs=-1,
    verbose=2,
    error_score='raise'
)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print(f"Best parameters: {grid_search.best_params_}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"R^2:{r2_score(y_test, y_pred):.2f}")

plt.figure(figsize=(8, 4))
plt.barh(features, best_model.feature_importances_)
plt.title('Feature importances')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()

test_with_preds = test.copy()
test_with_preds['y_pred'] = y_pred
weekly_comparison = create_weekly_agg(
    test_with_preds, 
    ['england_wales_demand', 'y_pred']
).tail(26)

plt.figure(figsize=(14, 6))
plt.plot(weekly_comparison['year_week'], weekly_comparison['england_wales_demand'], 
         label='Actual', marker='o', linestyle='-', color='blue')
plt.plot(weekly_comparison['year_week'], weekly_comparison['y_pred'], 
         label='Predicted', marker='x', linestyle='--', color='red')
plt.title('Weekly Demand Prediction vs Actual (Last 6 Months)')
plt.xlabel('Year-Week')
plt.ylabel('Demand (MW)')
plt.xticks(rotation=90, fontsize=8)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'max_depth': 5,
    'eta': 0.1,
    'tree_method': 'hist',  
    'eval_metric': 'rmse'
}

evals = [(dtrain, 'train'), (dtest, 'test')]

bst = xgb.train(
    params,
    dtrain,
    num_boost_round=200,
    obj=mse_cosine_obj,
    evals=evals,
    verbose_eval=10
)
y_Pred = bst.predict(dtest)

print("MSE:", mean_squared_error(y_test, y_Pred))
print("MAE:", mean_absolute_error(y_test, y_Pred))
print("R^2:", r2_score(y_test, y_Pred))

test_with_preds1 = test.copy()
test_with_preds1['y_pred'] = y_pred
test_with_preds1['y_Pred'] = y_Pred

weekly_comparison = create_weekly_agg(
    test_with_preds1, 
    ['england_wales_demand', 'y_pred', 'y_Pred']
).tail(26)

plt.figure(figsize=(14, 6))
plt.plot(weekly_comparison['year_week'], weekly_comparison['england_wales_demand'], 
         label='Actual', marker='o', linestyle='-', color='blue')
plt.plot(weekly_comparison['year_week'], weekly_comparison['y_pred'], 
         label='Predicted (Standard)', marker='x', linestyle='--', color='red')
plt.plot(weekly_comparison['year_week'], weekly_comparison['y_Pred'], 
         label='Predicted (Custom)', marker='s', linestyle='-.', color='green')
plt.title('Weekly Demand Prediction vs Actual (Last 6 Months)')
plt.xlabel('Year-Week')
plt.ylabel('Demand (MW)')
plt.xticks(rotation=90, fontsize=8)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

from sklearn.metrics.pairwise import cosine_similarity
print("cosine_similarit after MSE loss func", cosine_similarity([y_test], [y_pred]))
print("cosine_similarit after MSE+cos loss func :", cosine_similarity([y_test], [y_Pred]))
