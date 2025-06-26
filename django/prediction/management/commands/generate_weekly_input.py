import pandas as pd
import numpy as np
from datetime import datetime
import os
from django.core.management.base import BaseCommand
from django.conf import settings

# 您的实际数据源路径
DATA_SOURCE_PATH = r'D:\qiyezijin_root\xgb\模拟企业资金收入流水_近一年.csv'

def load_holiday_dates(year):
    """加载节假日日期（根据您的业务需求定制）"""
    return [
        datetime(year, 1, 1),   # 元旦
        datetime(year, 2, 10) if year == 2025 else datetime(year, 2, 1),  # 春节
        datetime(year, 4, 5),   # 清明节
        datetime(year, 5, 1),   # 劳动节
        datetime(year, 10, 1)   # 国庆节
    ]

def create_weekly_features(df):
    """
    创建周度特征（与训练时完全一致）
    
    参数:
        df: 包含日期和收入列的日度DataFrame
        
    返回:
        包含所有特征的周度DataFrame
    """
    # 按周聚合（周一为一周开始）
    df['WeekStart'] = df['日期'].dt.to_period('W-MON').apply(lambda r: r.start_time)
    weekly = df.groupby('WeekStart')['收入'].sum().reset_index()
    
    # 创建滞后特征（1-6周）
    for lag in range(1, 7):
        weekly[f'lag_{lag}'] = weekly['收入'].shift(lag)
    
    # 滚动平均特征
    weekly['rolling_mean_3'] = weekly['收入'].rolling(window=3).mean().shift(1)
    weekly['rolling_mean_6'] = weekly['收入'].rolling(window=6).mean().shift(1)
    
    # 时间特征
    weekly['week_of_year'] = weekly['WeekStart'].dt.isocalendar().week
    weekly['quarter'] = weekly['WeekStart'].dt.quarter
    
    # 周期性编码
    max_week = 52
    weekly['week_sin'] = np.sin(2 * np.pi * weekly['week_of_year'] / max_week)
    weekly['week_cos'] = np.cos(2 * np.pi * weekly['week_of_year'] / max_week)
    
    # 收入增长率（避免除零）
    weekly['income_growth_rate'] = weekly['收入'] / (weekly['lag_1'] + 1e-6) - 1
    
    # 节假日特征
    year = weekly['WeekStart'].dt.year.iloc[0]
    holiday_dates = load_holiday_dates(year)
    weekly['is_holiday_week'] = weekly['WeekStart'].apply(
        lambda x: 1 if any((date >= x) & (date < x + pd.Timedelta(days=7)) for date in holiday_dates) else 0
    )
    
    return weekly

class Command(BaseCommand):
    help = '生成周度预测输入数据'
    
    def handle(self, *args, **options):
        try:
            # 从实际数据源加载数据
            daily_df = pd.read_csv(DATA_SOURCE_PATH, encoding='GBK')  # 使用GBK编码读取中文文件
            
            # 重命名列以匹配处理逻辑
            daily_df = daily_df.rename(columns={
                '日期': '日期',
                '收入': '收入'
            })
            
            # 转换日期格式
            daily_df['日期'] = pd.to_datetime(daily_df['日期'])
            
            # 创建特征
            weekly_df = create_weekly_features(daily_df)
            
            # 获取最近一周的数据
            latest_week = weekly_df.iloc[[-1]]
            
            # 选择需要的特征
            feature_cols = [
                'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6',
                'week_of_year', 'rolling_mean_3', 'rolling_mean_6',
                'is_holiday_week', 'income_growth_rate', 'quarter',
                'week_sin', 'week_cos'
            ]
            
            # 保存到Django项目的data目录
            output_path = os.path.join(settings.BASE_DIR, 'prediction', 'data', 'weekly_input.csv')
            latest_week[feature_cols].to_csv(output_path, index=False, encoding='utf-8-sig')  # 使用BOM以支持Excel打开
            
            self.stdout.write(self.style.SUCCESS(f'周度预测输入文件已更新: {output_path}'))
            
            # 打印生成的数据预览
            self.stdout.write(self.style.SUCCESS('生成的数据预览:'))
            self.stdout.write(str(latest_week[feature_cols].head()))
            
        except Exception as e:
            import traceback
            self.stdout.write(self.style.ERROR(f'生成周度输入数据失败: {str(e)}'))
            self.stdout.write(self.style.ERROR(traceback.format_exc()))