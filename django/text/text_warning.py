import os
import sys

# 获取项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 将项目根目录添加到 Python 搜索路径
sys.path.append(BASE_DIR)

# 设置 DJANGO_SETTINGS_MODULE 环境变量
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproject2.settings')

# 导入 Django 并设置环境
import django
django.setup()

from prediction.models import ForecastRecord
from prediction.alert import XGBModelPredictor, check_and_alert

# 单例模式实例化预测器
predictor = XGBModelPredictor()

def test_weekly_alert():
    # 模拟周度预测
    result = predictor.predict_weekly_income()
    print("周度预测结果:", result)

    if result["status"] == "success":
        # 获取最新的周度预测记录
        try:
            latest_record = ForecastRecord.objects.filter(
                forecast_type='weekly'
            ).latest('prediction_date')
            print("最新周度预测记录:", latest_record)

            # 验证预警触发状态
            print("预警是否触发:", result["alert_triggered"])

            # 验证数据库更新
            latest_record.refresh_from_db()
            print("数据库中的预警状态:", latest_record.is_alerted)
            print("数据库中的预警偏差:", latest_record.alert_deviation)

        except ForecastRecord.DoesNotExist:
            print("无周度历史预测记录")

def test_monthly_alert():
    # 模拟月度预测
    result = predictor.predict_monthly_income()
    print("月度预测结果:", result)

    if result["status"] == "success":
        # 获取最新的月度预测记录
        try:
            latest_record = ForecastRecord.objects.filter(
                forecast_type='monthly'
            ).latest('prediction_date')
            print("最新月度预测记录:", latest_record)

            # 验证预警触发状态
            print("预警是否触发:", result["alert_triggered"])

            # 验证数据库更新
            latest_record.refresh_from_db()
            print("数据库中的预警状态:", latest_record.is_alerted)
            print("数据库中的预警偏差:", latest_record.alert_deviation)

        except ForecastRecord.DoesNotExist:
            print("无月度历史预测记录")

if __name__ == "__main__":
    test_weekly_alert()
    print("-" * 50)
    test_monthly_alert()