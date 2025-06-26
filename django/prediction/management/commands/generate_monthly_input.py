import pandas as pd
import numpy as np
import os
import datetime
from django.core.management.base import BaseCommand
from django.conf import settings

DATA_SOURCE_PATH = r'D:\qiyezijin_root\xgb\模拟企业资金收入流水_近一年.csv'

def create_prediction_features(df):
    """
    创建用于预测的特征（仅使用历史数据）
    注意：此函数不包含当前月的数据，仅使用历史数据生成预测所需的特征
    """
    # 确保日期列正确转换并排序
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values('日期')
    
    # 按月聚合（历史数据）
    df['MonthStart'] = df['日期'].dt.to_period('M').dt.start_time
    monthly = df.groupby('MonthStart', as_index=False)['收入'].sum()
    
    # 仅使用历史数据（排除当前月）
    # 获取最后完整月份的日期
    last_complete_month = monthly['MonthStart'].max()
    historical = monthly[monthly['MonthStart'] < last_complete_month]
    
    # 重新索引以确保连续月份
    historical = historical.set_index('MonthStart').asfreq('MS').reset_index()
    
    # 填充缺失的收入值为0
    historical['收入'] = historical['收入'].fillna(0)
    
    # 计算滞后特征（使用历史收入）
    historical['lag_1'] = historical['收入'].shift(1)
    historical['lag_2'] = historical['收入'].shift(2)
    historical['lag_3'] = historical['收入'].shift(3)
    
    # 计算收入增长率（使用历史数据）
    historical['income_growth_rate'] = historical.apply(
        lambda row: (row['lag_1'] / row['lag_2'] - 1) if (row['lag_2'] > 0 and not pd.isna(row['lag_2'])) else 0,
        axis=1
    )
    
    # 滚动特征（使用历史数据）
    historical['rolling_mean_2'] = historical['收入'].shift(1).rolling(window=2, min_periods=1).mean()
    historical['rolling_mean_3'] = historical['收入'].shift(1).rolling(window=3, min_periods=1).mean()
    
    # 时间特征（预测月份 = 最后完整月份的下一个月）
    prediction_month = last_complete_month + pd.DateOffset(months=1)
    
    # 创建预测特征行
    prediction_features = pd.DataFrame({
        'MonthStart': [prediction_month],
        'month': [prediction_month.month],
        'quarter': [(prediction_month.month-1)//3 + 1],
        'year': [prediction_month.year],
        'lag_1': [historical['收入'].iloc[-1]],  # 最后一个月的历史收入
        'lag_2': [historical['收入'].iloc[-2]] if len(historical) >= 2 else historical['收入'].mean(),
        'lag_3': [historical['收入'].iloc[-3]] if len(historical) >= 3 else historical['收入'].mean(),
        'income_growth_rate': [historical['income_growth_rate'].iloc[-1]] if len(historical) >= 2 else 0,
        'rolling_mean_2': [historical['收入'].iloc[-2:].mean()] if len(historical) >= 2 else historical['收入'].mean(),
        'rolling_mean_3': [historical['收入'].iloc[-3:].mean()] if len(historical) >= 3 else historical['收入'].mean(),
    })
    
    # 周期性特征
    prediction_features['month_sin'] = np.sin(2 * np.pi * prediction_features['month'] / 12)
    prediction_features['month_cos'] = np.cos(2 * np.pi * prediction_features['month'] / 12)
    
    # 节假日特征
    prediction_features['is_holiday_month'] = prediction_features['month'].apply(
        lambda x: 1 if x in [1, 5, 10] else 0
    )
    
    return prediction_features

class Command(BaseCommand):
    help = '生成月度预测输入数据'
    
    def handle(self, *args, **options):
        try:
            self.stdout.write(self.style.SUCCESS(f"开始生成月度预测输入数据: {datetime.datetime.now()}"))
            
            # 1. 验证数据源
            if not os.path.exists(DATA_SOURCE_PATH):
                raise FileNotFoundError(f"数据源文件不存在: {DATA_SOURCE_PATH}")
            
            # 2. 加载数据
            self.stdout.write("加载数据源...")
            daily_df = pd.read_csv(DATA_SOURCE_PATH, encoding='GBK')
            
            # 3. 验证数据列
            required_columns = {'日期', '收入'}
            if not required_columns.issubset(daily_df.columns):
                missing = required_columns - set(daily_df.columns)
                raise ValueError(f"数据源缺少必要列: {', '.join(missing)}")
            
            # 4. 重命名列
            daily_df = daily_df.rename(columns={'日期': '日期', '收入': '收入'})
            
            # 5. 创建预测特征
            self.stdout.write("创建预测特征...")
            prediction_df = create_prediction_features(daily_df)
            
            # 6. 定义特征列
            feature_cols = [
                'lag_1', 'lag_2', 'lag_3',
                'month', 'quarter', 'year',
                'rolling_mean_2', 'rolling_mean_3',
                'is_holiday_month', 'income_growth_rate',
                'month_sin', 'month_cos'
            ]
            
            # 7. 验证特征完整性
            missing_features = [col for col in feature_cols if col not in prediction_df.columns]
            if missing_features:
                raise ValueError(f"缺失特征列: {', '.join(missing_features)}")
            
            # 8. 保存数据
            output_dir = os.path.join(settings.BASE_DIR, 'prediction', 'data')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, 'monthly_input.csv')
            
            prediction_df[feature_cols].to_csv(output_path, index=False, encoding='utf-8-sig')
            
            # 9. 详细日志输出
            self.stdout.write(self.style.SUCCESS(f"预测输入文件已保存: {output_path}"))
            self.stdout.write(self.style.SUCCESS("预测特征值:"))
            
            for col in feature_cols:
                value = prediction_df[col].values[0]
                self.stdout.write(f"  {col}: {value}")
            
            # 10. 特别验证income_growth_rate
            if 'income_growth_rate' in prediction_df.columns:
                igr = prediction_df['income_growth_rate'].values[0]
                self.stdout.write(self.style.SUCCESS(f"income_growth_rate 值: {igr}"))
            else:
                self.stdout.write(self.style.ERROR("income_growth_rate 特征缺失!"))
            
            return True
            
        except Exception as e:
            import traceback
            error_msg = f"生成月度输入数据失败: {str(e)}"
            self.stdout.write(self.style.ERROR(error_msg))
            tb = traceback.format_exc()
            self.stdout.write(self.style.ERROR(tb))
            
            # 创建详细的错误日志
            error_dir = os.path.join(settings.BASE_DIR, 'prediction', 'errors')
            os.makedirs(error_dir, exist_ok=True)
            error_file = os.path.join(error_dir, f"monthly_input_error_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
            
            with open(error_file, 'w', encoding='utf-8') as f:
                f.write(f"错误时间: {datetime.datetime.now()}\n")
                f.write(f"错误信息: {error_msg}\n")
                f.write("追踪信息:\n")
                f.write(tb)
                f.write("\n\n数据源信息:\n")
                f.write(f"路径: {DATA_SOURCE_PATH}\n")
                
                if 'daily_df' in locals():
                    f.write(f"数据行数: {len(daily_df)}\n")
                    f.write(f"数据列: {', '.join(daily_df.columns)}\n")
                    if '日期' in daily_df.columns:
                        f.write(f"日期范围: {daily_df['日期'].min()} 至 {daily_df['日期'].max()}\n")
            
            self.stdout.write(self.style.ERROR(f"详细错误日志已保存: {error_file}"))
            return False