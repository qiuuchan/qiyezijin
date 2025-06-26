import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
import matplotlib.pyplot as plt
import joblib
import os
from pathlib import Path
from pandas.tseries.offsets import DateOffset

# --- 中文字体设置 ---
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False    # 负号正常显示

def load_holiday_dates(year):
    """动态加载指定年份的节假日数据（支持从文件或使用预设值）"""
    try:
        # 优先从脚本所在目录加载节假日文件
        script_dir = Path(__file__).parent
        holiday_file = script_dir / f'holidays_{year}.csv'
        
        if holiday_file.exists():
            holidays = pd.read_csv(holiday_file, parse_dates=['date'])
            return holidays['date'].tolist()
        
        # 若文件不存在，使用预设节假日（支持跨年份，春节日期需手动调整）
        start_date = pd.Timestamp(f'{year}-01-01')
        end_date = pd.Timestamp(f'{year}-12-31')
        default_holidays = [
            # 元旦
            pd.Timestamp(f'{year}-01-01'),
            # 春节（示例日期，实际需根据农历调整）
            pd.Timestamp(f'{year}-02-11') if year == 2025 else None,
            # 清明节
            pd.Timestamp(f'{year}-04-05'),
            # 劳动节
            pd.Timestamp(f'{year}-05-01'),
            # 国庆节
            pd.Timestamp(f'{year}-10-01')
        ]
        return [d for d in default_holidays if d is not None and start_date <= d <= end_date]
    except Exception as e:
        print(f"加载节假日数据失败: {e}")
        return []

def handle_outliers(df, column, method='clip', iqr_multiplier=2.0):
    """
    异常值处理函数（增强版，保留更多原始信息）
    method: 'clip' 温和截断, 'log' 对数变换, 'interpolate' 插值
    """
    try:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR
        
        print(f"异常值检测[{column}]：下界={lower_bound:.2f}，上界={upper_bound:.2f}")
        
        if method == 'clip':
            # 更温和的截断：边界值扩展20%，避免过度修剪
            df[column] = df[column].apply(lambda x: 
                                          lower_bound * 0.8 if x < lower_bound else 
                                          (upper_bound * 1.2 if x > upper_bound else x))
        elif method == 'log':
            # 对数变换（处理正偏态数据）
            df[column] = np.log1p(df[column])
        elif method == 'interpolate':
            # 线性插值（适用于少量异常值）
            mask = (df[column] < lower_bound) | (df[column] > upper_bound)
            df.loc[mask, column] = df[column].interpolate()
        
        num_outliers = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
        print(f"处理后异常值数量：{num_outliers}")
        return df
    except Exception as e:
        print(f"异常值处理失败: {e}")
        return df

def create_features(df):
    """特征工程函数（整合所有特征构造逻辑）"""
    # 1. 基础时间特征
    df['month'] = df['ds'].dt.month
    df['quarter'] = df['ds'].dt.quarter
    df['year'] = df['ds'].dt.year
    
    # 2. 滞后特征（1-3个月）
    for lag in range(1, 4):
        df[f'lag_{lag}'] = df['y'].shift(lag)
    
    # 3. 滚动平均特征（2个月、3个月）
    df['rolling_mean_2'] = df['y'].rolling(window=2).mean().shift(1)
    df['rolling_mean_3'] = df['y'].rolling(window=3).mean().shift(1)
    
    # 4. 节假日特征（动态加载）
    year = df['ds'].dt.year.iloc[0]
    holiday_dates = load_holiday_dates(year)
    df['is_holiday_month'] = df['ds'].apply(
        lambda x: 1 if any((date >= x) & (date < x + pd.offsets.MonthEnd()) for date in holiday_dates) else 0
    )
    
    # 5. 收入增长率（避免除零错误）
    df['income_growth_rate'] = df['y'] / (df['lag_1'] + 1e-6)
    
    # 6. 月份周期性编码（正弦余弦变换）
    max_month = 12
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / max_month)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / max_month)
    
    return df

def train_xgb(df: pd.DataFrame, test_size=1):
    """训练XGBoost模型并评估"""
    train_df = df.iloc[:-test_size]
    test_df = df.iloc[-test_size:]

    feature_cols = [f'lag_{i}' for i in range(1, 4)] + [
        'month', 'quarter', 'year', 'rolling_mean_2', 'rolling_mean_3', 
        'is_holiday_month', 'income_growth_rate', 'month_sin', 'month_cos'
    ]
    X_train, y_train = train_df[feature_cols], train_df['y']
    X_test, y_test = test_df[feature_cols], test_df['y']

    model = xgb.XGBRegressor(random_state=42)
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'gamma': [0, 0.5, 1],
        'min_child_weight': [1, 3, 5]
    }

    # 时间序列交叉验证调参
    print("开始时间序列交叉验证调参...")
    ts_cv = TimeSeriesSplit(n_splits=3, test_size=1)  # 3折交叉验证，每次测试集1个月
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=30,
        scoring='neg_mean_absolute_percentage_error',
        cv=ts_cv,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)

    print("最佳参数组合:", random_search.best_params_)

    # 用最佳参数重新训练
    best_model = random_search.best_estimator_

    y_pred = best_model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100

    print("\n===== XGBoost 月度预测结果 =====")
    for i, date in enumerate(test_df['ds'].dt.date):
        print(f"预测月份： {date}")
        print(f"预测值：   {y_pred[i]:.2f}")
        print(f"真实值：   {y_test.values[i]:.2f}")
        print(f"误差：     {abs(y_pred[i] - y_test.values[i]):.2f}")
    print(f"平均绝对百分比误差(MAPE)： {mape:.2f}%")

    print("\n特征重要性:")
    importance = best_model.feature_importances_
    for feat, imp in zip(feature_cols, importance):
        print(f"{feat}: {imp:.3f}")

    # 保存最佳模型（关键修改：指定统一保存路径）
    model_dir = Path(r"D:\qiyezijin_root\django\prediction\models")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / 'xgb_monthly_model.pkl'
    joblib.dump(best_model, model_path)
    print(f"\n模型已保存至: {model_path}")

    return best_model, X_test, y_test, y_pred

def plot_results(df: pd.DataFrame, model):
    """绘制历史数据和预测结果"""
    all_df = create_features(df.copy())
    all_df = all_df.dropna().reset_index(drop=True)
    
    # 预测未来1个月
    last_date = all_df['ds'].iloc[-1]
    next_date = last_date + DateOffset(months=1)
    
    # 准备预测特征
    feature_cols = [f'lag_{i}' for i in range(1, 4)] + [
        'month', 'quarter', 'year', 'rolling_mean_2', 'rolling_mean_3', 
        'is_holiday_month', 'income_growth_rate', 'month_sin', 'month_cos'
    ]
    
    # 创建预测行
    pred_row = all_df.iloc[-1:].copy()
    pred_row['ds'] = next_date
    pred_row['month'] = next_date.month
    pred_row['quarter'] = next_date.quarter
    pred_row['year'] = next_date.year
    
    # 更新滞后特征和滚动特征
    for lag in range(1, 4):
        if lag == 1:
            pred_row[f'lag_{lag}'] = all_df['y'].iloc[-1]
        else:
            pred_row[f'lag_{lag}'] = all_df[f'lag_{lag-1}'].iloc[-1]
    
    pred_row['rolling_mean_2'] = (all_df['y'].iloc[-1] + all_df['y'].iloc[-2]) / 2
    pred_row['rolling_mean_3'] = (all_df['y'].iloc[-1] + all_df['y'].iloc[-2] + all_df['y'].iloc[-3]) / 3
    
    # 更新节假日特征
    year = next_date.year
    holiday_dates = load_holiday_dates(year)
    pred_row['is_holiday_month'] = 1 if any(
        (date >= next_date) & (date < next_date + pd.offsets.MonthEnd()) for date in holiday_dates
    ) else 0
    
    # 更新增长率
    pred_row['income_growth_rate'] = pred_row['lag_1'] / (all_df['lag_1'].iloc[-1] + 1e-6)
    
    # 更新周期性编码
    max_month = 12
    pred_row['month_sin'] = np.sin(2 * np.pi * next_date.month / max_month)
    pred_row['month_cos'] = np.cos(2 * np.pi * next_date.month / max_month)
    
    # 预测
    y_pred = model.predict(pred_row[feature_cols])[0]
    
    # 可视化
    plt.figure(figsize=(12, 6))
    plt.plot(df['ds'], df['y'], 'o-', label='历史数据', color='blue', alpha=0.7)
    plt.scatter([next_date], [y_pred], color='red', s=150, marker='*', label='预测值')
    plt.annotate(f'{y_pred:.2f}', 
                 (next_date, y_pred), 
                 textcoords="offset points", 
                 xytext=(0, 15), 
                 ha='center',
                 fontsize=12)
    
    # 标记节假日月份
    holiday_months = all_df[all_df['is_holiday_month'] == 1]['ds']
    for month in holiday_months:
        plt.axvline(x=month, color='green', linestyle='--', alpha=0.3)
    
    plt.title('月度收入预测（含节假日标记）')
    plt.xlabel('日期')
    plt.ylabel('收入金额')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('monthly_prediction.png', dpi=300)
    plt.show()

def main():
    """主函数（整合数据处理、建模、评估全流程）"""
    try:
        # --- 数据加载模块 ---
        script_dir = Path(__file__).parent
        
        # 尝试多种路径查找方式
        possible_paths = [
            script_dir / 'monthly.csv',          # 同目录
            script_dir.parent / 'monthly.csv',   # 上一级目录
            Path(os.environ.get('USERPROFILE')) / 'monthly.csv',  # 用户目录
            Path.cwd() / 'monthly.csv'           # 当前工作目录
        ]
        
        data_path = None
        for path in possible_paths:
            if path.exists():
                data_path = path
                break
        
        if data_path is None:
            # 若所有路径都找不到，提供交互式文件选择
            import tkinter as tk
            from tkinter import filedialog
            
            root = tk.Tk()
            root.withdraw()
            print("未找到数据文件，正在打开文件选择器...")
            data_path = Path(filedialog.askopenfilename(
                title="选择月度收入数据文件",
                filetypes=[("CSV文件", "*.csv")]
            ))
            root.destroy()
            
            if not data_path.exists():
                print("未选择有效文件，程序退出")
                return
        
        print(f"正在从 {data_path} 加载数据...")
        df = pd.read_csv(data_path, parse_dates=['ds'])
        df = df.sort_values('ds').reset_index(drop=True)
        print(f"数据加载完成，行数: {len(df)}, 列数: {len(df.columns)}")
        
        # --- 数据处理模块 ---
        # 异常值处理
        df = handle_outliers(df, 'y', method='clip', iqr_multiplier=2.0)
        
        # 特征工程
        df_features = create_features(df)
        
        # 过滤缺失值
        df_features = df_features.dropna().reset_index(drop=True)
        print(f"特征构造完成，有效月数: {len(df_features)}")
        
        # --- 建模与评估模块 ---
        # 训练模型（使用最后1个月作为测试集）
        model, X_test, y_test, y_pred = train_xgb(df_features, test_size=1)
        
        # 可视化预测结果
        plot_results(df, model)
        
    except Exception as e:
        print(f"程序运行出错: {e}")
        # 打印详细异常信息（便于调试）
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()