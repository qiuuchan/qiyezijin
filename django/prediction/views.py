from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
import json
from datetime import datetime

# 为周度预测创建完整数据结构
def get_weekly_prediction():
    labels = ['第1周', '第2周', '第3周', '第4周', '第5周', '第6周', '第7周', '第8周', '第9周', '第10周', '第11周', '第12周']
    predicted = [125000, 132000, 128000, 145000, 138000, 152000, 149000, 165000, 172000, 168000, 185000, 192000]
    actual = [128000, 130000, 127000, 142000, 140000, 150000, 151000, 162000, 170000, 171000, 182000, None]
    
    return {
        'status': 'success',
        'data': {
            'this_week': {
                'income': predicted[-1],
                'growth': 3.8,
                'growth_value': 7000
            },
            'error_rate': 4.23,
            'period': '2025Q2-Q3',
            'period_start': '2025-04-01',
            'period_end': '2025-06-30',
            'completed_weeks': 8,
            'predicting_weeks': 4,
            'weeks': [{
                'week_number': f'第{i+1}周',
                'start_date': f'2025-04-{i*7+1:02d}',
                'predicted_income': predicted[i],
                'actual_income': actual[i],
                'error_rate': round(abs((predicted[i] - (actual[i] or predicted[i])) / (actual[i] or predicted[i]) * 100), 2) if i < len(actual) - 1 else 0,
                'trend': '增长' if predicted[i] > (predicted[i-1] if i > 0 else 0) else '下降' if i > 0 else '持平'
            } for i in range(len(labels))],
            'error_distribution': [15, 25, 30, 20, 10],
            'accuracy_trend': {
                'labels': labels[:8],
                'data': [5.2, 4.8, 4.5, 4.2, 4.0, 3.8, 3.9, 4.0]
            }
        }
    }

# 为月度预测创建完整数据结构
def get_monthly_prediction():
    labels = ['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月']
    predicted = [520000, 580000, 620000, 680000, 750000, 820000, 900000, 980000, 1050000, 1120000, 1200000, 1280000]
    actual = [510000, 575000, 610000, 672000, 745000, 812000, None, None, None, None, None, None]
    
    # 修正历史精度计算逻辑（使用实际误差比例）
    history_accuracy = []
    for i in range(min(len(actual), 8)):
        if actual[i] and predicted[i]:
            error_rate = abs((predicted[i] - actual[i]) / predicted[i] * 100)
            accuracy = 100 - error_rate
        else:
            accuracy = 0
        history_accuracy.append({
            'month': labels[i],
            'accuracy': round(accuracy, 1)  # 保留一位小数
        })
    
    return {
        'status': 'success',
        'data': {
            'history_months': 12,
            'predict_months': 6,
            'avg_error': round(sum(abs(predicted[i] - (actual[i] or predicted[i])) / (actual[i] or predicted[i]) * 100 for i in range(6)) / 6, 2),
            'next_month': {
                'label': '下月预测收入',
                'income': predicted[datetime.now().month % 12],
                'growth': 6.7
            },
            'confidence': 86.5,
            'history_error': {'label': '历史平均误差'},
            'better_than_industry': 83,
            'best_worst': {
                'label_best': '最佳月份',
                'label_worst': '最差月份'
            },
            'best_month': '2025年3月',
            'worst_month': '2024年11月',
            'seasonal': {
                'label': '季节性影响',
                'peak': 'Q3 为高峰季',
                'tip': '注意9月回款高峰',
                'q1_growth': 3.2,
                'q3_growth': 6.7
            },
            'trend_labels': labels,
            'predicted_income': predicted,
            'actual_income': actual,
            'forecast_details': [{
                'month': labels[i],
                'income': predicted[i],
                'is_peak': i in [6, 7, 8]  # Q3月份
            } for i in range(len(labels))],
            'history_accuracy': history_accuracy  # 修正后的数据
        }
    }

def weekly_forecast(request):
    if request.method == 'GET':
        result = get_weekly_prediction()
        if result.get("status") == "error":
            return JsonResponse(result, status=500)
        else:
            # 直接传递数据字典而不是JSON字符串
            return render(request, 'prediction/weekly.html', {
                'weekly_forecast': result['data']
            })
    else:
        return JsonResponse({"error": "仅支持 GET 请求"}, status=405)

def monthly_forecast(request):
    if request.method == 'GET':
        result = get_monthly_prediction()
        if result.get("status") == "error":
            return JsonResponse(result, status=500)
        else:
            return render(request, 'prediction/monthly.html', {
                'monthly_forecast': result['data']
            })
    else:
        return JsonResponse({"error": "仅支持 GET 请求"}, status=405)

# 其他视图保持不变...

def home(request):
    # 可传递动态数据（如预测精度、历史趋势等）
    context = {
        'weekly_accuracy': 4.23,
        'monthly_accuracy': 0.72,
    }
    return render(request, 'home.html', context)

def dashboard(request):
    return render(request, 'dashboard.html')