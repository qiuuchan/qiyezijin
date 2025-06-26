import os
import pandas as pd
import joblib
from django.conf import settings
from datetime import datetime, timedelta

class XGBModelPredictor:
    """XGBoost模型预测器，支持周度和月度收入预测"""
    
    def __init__(self):
        # 模型路径配置
        self.weekly_model_path = os.path.join(settings.BASE_DIR, 'prediction', 'models', 'xgb_weekly_model.pkl')
        self.monthly_model_path = os.path.join(settings.BASE_DIR, 'prediction', 'models', 'xgb_monthly_model.pkl')
        
        # 输入数据路径配置
        self.weekly_input_path = os.path.join(settings.BASE_DIR, 'prediction', 'data', 'weekly_input.csv')
        self.monthly_input_path = os.path.join(settings.BASE_DIR, 'prediction', 'data', 'monthly_input.csv')
        
        # 模型缓存
        self.weekly_model = None
        self.monthly_model = None
    
    def load_weekly_model(self):
        """加载周度预测模型"""
        if self.weekly_model is None:
            try:
                self.weekly_model = joblib.load(self.weekly_model_path)
                print(f"✅ 周度模型加载成功: {self.weekly_model_path}")
            except FileNotFoundError:
                print(f"❌ 周度模型文件不存在: {self.weekly_model_path}")
                return None
        return self.weekly_model
    
    def load_monthly_model(self):
        """加载月度预测模型"""
        if self.monthly_model is None:
            try:
                self.monthly_model = joblib.load(self.monthly_model_path)
                print(f"✅ 月度模型加载成功: {self.monthly_model_path}")
            except FileNotFoundError:
                print(f"❌ 月度模型文件不存在: {self.monthly_model_path}")
                return None
        return self.monthly_model
    
    def prepare_weekly_input_data(self, custom_data=None):
        """准备周度预测输入数据
        
        Args:
            custom_data: 自定义输入数据，用于覆盖默认输入文件
        
        Returns:
            DataFrame: 处理后的输入数据
        """
        if custom_data:
            # 使用自定义数据
            df = pd.DataFrame([custom_data])
        else:
            # 从CSV文件加载
            try:
                df = pd.read_csv(self.weekly_input_path)
            except FileNotFoundError:
                print(f"❌ 周度输入数据不存在: {self.weekly_input_path}")
                return None
        
        # 确保所有必要特征都存在（更新为与训练模型一致的特征集）
        required_features = [
            'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6',
            'week_of_year', 'rolling_mean_3', 'rolling_mean_6',
            'is_holiday_week', 'income_growth_rate', 'quarter',
            'week_sin', 'week_cos'
        ]
        
        missing_features = [f for f in required_features if f not in df.columns]
        if missing_features:
            print(f"❌ 输入数据缺少必要特征: {missing_features}")
            return None
            
        return df[required_features]
    
    def prepare_monthly_input_data(self, custom_data=None):
        """准备月度预测输入数据"""
        if custom_data:
            df = pd.DataFrame([custom_data])
        else:
            try:
                df = pd.read_csv(self.monthly_input_path)
            except FileNotFoundError:
                print(f"❌ 月度输入数据不存在: {self.monthly_input_path}")
                return None
        
        # 确保所有必要特征都存在
        required_features = [
            'lag_1', 'lag_2', 'lag_3',
            'month', 'quarter', 'year',
            'rolling_mean_2', 'rolling_mean_3',
            'is_holiday_month', 'income_growth_rate',
            'month_sin', 'month_cos'
        ]
        
        missing_features = [f for f in required_features if f not in df.columns]
        if missing_features:
            print(f"❌ 输入数据缺少必要特征: {missing_features}")
            return None
            
        return df[required_features]
    
    def predict_weekly_income(self, custom_data=None):
        """预测周度收入
        
        Args:
            custom_data: 自定义输入数据
            
        Returns:
            dict: 包含预测结果的字典
        """
        model = self.load_weekly_model()
        if model is None:
            return {"status": "error", "message": "周度模型加载失败"}
            
        input_data = self.prepare_weekly_input_data(custom_data)
        if input_data is None:
            return {"status": "error", "message": "周度输入数据准备失败"}
            
        try:
            prediction = model.predict(input_data)[0]
            
            # 计算预测日期范围（当前日期所在周的周一至周日）
            today = datetime.now()
            monday = today - timedelta(days=today.weekday())
            sunday = monday + timedelta(days=6)
            
            return {
                "predicted_income": float(prediction),
                "prediction_date": today.strftime("%Y-%m-%d"),
                "prediction_week_range": f"{monday.strftime('%Y-%m-%d')} 至 {sunday.strftime('%Y-%m-%d')}",
                "model_version": "v1.0",
                "status": "success"
            }
        except Exception as e:
            return {"status": "error", "message": f"周度预测出错: {str(e)}"}
    
    def predict_monthly_income(self, custom_data=None):
        """预测月度收入"""
        model = self.load_monthly_model()
        if model is None:
            return {"status": "error", "message": "月度模型加载失败"}
            
        input_data = self.prepare_monthly_input_data(custom_data)
        if input_data is None:
            return {"status": "error", "message": "月度输入数据准备失败"}
            
        try:
            prediction = model.predict(input_data)[0]
            
            # 获取当前月份
            today = datetime.now()
            month_name = today.strftime("%B")
            year = today.year
            
            return {
                "predicted_income": float(prediction),
                "prediction_date": today.strftime("%Y-%m-%d"),
                "prediction_month": f"{month_name} {year}",
                "model_version": "v1.0",
                "status": "success"
            }
        except Exception as e:
            return {"status": "error", "message": f"月度预测出错: {str(e)}"}

# 单例模式实例化预测器
predictor = XGBModelPredictor()

# 便捷函数，用于快速调用周度预测
def get_weekly_prediction(custom_data=None):
    return predictor.predict_weekly_income(custom_data)

# 便捷函数，用于快速调用月度预测
def get_monthly_prediction(custom_data=None):
    return predictor.predict_monthly_income(custom_data)