import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
import matplotlib.pyplot as plt
import joblib
import os
from pathlib import Path
from datetime import datetime

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
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31)
        default_holidays = [
            # 元旦
            datetime(year, 1, 1),
            # 春节（示例日期，实际需根据农历调整）
            datetime(year, 2, 11) if year == 2025 else None,
            # 清明节
            datetime(year, 4, 5),
            # 劳动节
            datetime(year, 5, 1),
            # 国庆节
            datetime(year, 10, 1)
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
    df['week_of_year'] = df['WeekStart'].dt.isocalendar().week
    df['quarter'] = df['WeekStart'].dt.quarter
    
    # 2. 滞后特征（1-6周）
    for lag in range(1, 7):
        df[f'lag_{lag}'] = df['Income'].shift(lag)
    
    # 3. 滚动平均特征（3周、6周）
    df['rolling_mean_3'] = df['Income'].rolling(window=3).mean().shift(1)
    df['rolling_mean_6'] = df['Income'].rolling(window=6).mean().shift(1)
    
    # 4. 节假日特征（动态加载）
    year = df['WeekStart'].dt.year.iloc[0]
    holiday_dates = load_holiday_dates(year)
    df['is_holiday_week'] = df['WeekStart'].apply(
        lambda x: 1 if any((date >= x) & (date < x + pd.Timedelta(days=7)) for date in holiday_dates) else 0
    )
    
    # 5. 收入增长率（避免除零错误）
    df['income_growth_rate'] = df['Income'] / (df['lag_1'] + 1e-6)
    
    # 6. 周次周期性编码（正弦余弦变换）
    max_week = 52
    df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / max_week)
    df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / max_week)
    
    return df

def main():
    """主函数（整合数据处理、建模、评估全流程，指定模型保存路径）"""
    try:
        # --- 数据加载模块（增强路径查找逻辑）---
        script_dir = Path(__file__).parent  # 获取当前脚本所在目录
        
        # 尝试多种路径查找方式
        possible_paths = [
            script_dir / 'daily_income.csv',          # 同目录
            script_dir.parent / 'daily_income.csv',   # 上一级目录
            Path(os.environ.get('USERPROFILE')) / 'daily_income.csv',  # 用户目录
            Path.cwd() / 'daily_income.csv'           # 当前工作目录
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
                title="选择日度收入数据文件",
                filetypes=[("CSV文件", "*.csv")]
            ))
            root.destroy()
            
            if not data_path.exists():
                print("未选择有效文件，程序退出")
                return
        
        print(f"正在从 {data_path} 加载数据...")
        df = pd.read_csv(data_path, parse_dates=['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        print(f"数据加载完成，行数: {len(df)}, 列数: {len(df.columns)}")
        
        # --- 数据处理模块 ---
        # 按周聚合（周一为一周开始）
        df['WeekStart'] = df['Date'].dt.to_period('W-MON').apply(lambda r: r.start_time)
        weekly = df.groupby('WeekStart')['Income'].sum().reset_index()
        print(f"周度聚合完成，周数: {len(weekly)}")
        
        # 异常值处理（默认温和截断）
        weekly = handle_outliers(weekly, 'Income', method='clip', iqr_multiplier=2.0)
        
        # 特征工程
        weekly = create_features(weekly)
        
        # 过滤缺失值
        weekly = weekly.dropna().reset_index(drop=True)
        print(f"特征构造完成，有效周数: {len(weekly)}")
        
        # --- 建模与评估模块 ---
        # 划分训练集和测试集（最后8周作为测试集）
        test_size = 8
        train = weekly.iloc[:-test_size]
        test = weekly.iloc[-test_size:]
        print(f"数据集划分：训练集 {len(train)} 周，测试集 {len(test)} 周")
        
        # 准备特征和标签
        feature_cols = [f'lag_{i}' for i in range(1, 7)] + [
            'week_of_year', 'rolling_mean_3', 'rolling_mean_6', 
            'is_holiday_week', 'income_growth_rate', 'quarter', 'week_sin', 'week_cos'
        ]
        X_train, y_train = train[feature_cols], train['Income']
        X_test, y_test = test[feature_cols], test['Income']
        print(f"特征维度：{X_train.shape[1]} 维")
        
        # 定义XGBoost模型和参数搜索空间
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
        ts_cv = TimeSeriesSplit(n_splits=3, test_size=4)  # 3折交叉验证，每次测试集4周
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
        
        # 预测测试集
        y_pred = best_model.predict(X_test)
        
        # 计算评估指标
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        rmse = np.sqrt(((y_pred - y_test) ** 2).mean())
        
        print("===== XGBoost 周度预测结果（优化版） =====")
        for date, pred, true in zip(test['WeekStart'], y_pred, y_test):
            print(f"预测周起始：{date.date()}，预测值：{pred:.2f}，真实值：{true:.2f}，误差：{abs(pred-true):.2f}")
        print(f"调参后评估：MAPE={mape:.2f}%，RMSE={rmse:.2f}")
        
        # 特征重要性分析
        feature_importance = best_model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        print("\n特征重要性排名：")
        print(importance_df)
        
        # --- 可视化与模型保存 ---
        # 可视化
        plt.figure(figsize=(12, 7))
        plt.plot(weekly['WeekStart'], weekly['Income'], label='历史周度收入', marker='o', alpha=0.7)
        plt.plot(test['WeekStart'], y_pred, label='预测收入', marker='x', color='red', linewidth=2)
        
        # 标记节假日周
        holiday_weeks = weekly[weekly['is_holiday_week'] == 1]['WeekStart']
        for week in holiday_weeks:
            plt.axvline(x=week, color='green', linestyle='--', alpha=0.3)
        
        plt.title('周度资金收入预测（含动态特征与时间序列验证）')
        plt.xlabel('周起始日期')
        plt.ylabel('收入金额')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig('weekly_prediction.png', dpi=300)
        plt.show()
        
        # 保存最佳模型（关键修改：指定统一保存路径）
        model_dir = Path(r"D:\qiyezijin_root\django\prediction\models")
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / 'xgb_weekly_model.pkl'
        joblib.dump(best_model, model_path)
        print(f"最佳模型已保存至: {model_path}")
        
    except Exception as e:
        print(f"程序运行出错: {e}")
        # 打印详细异常信息（便于调试）
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()