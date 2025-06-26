from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),  # 根路径映射到 home 视图
    # 周度预测接口：/api/weekly/
    path('weekly/', views.weekly_forecast, name='weekly_forecast'),
    # 月度预测接口：/api/monthly/
    path('monthly/', views.monthly_forecast, name='monthly_forecast'),
    
    path('dashboard/', views.dashboard, name='dashboard'),
]