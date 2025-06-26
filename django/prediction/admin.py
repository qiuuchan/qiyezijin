from django.contrib import admin

# Register your models here.
# prediction/admin.py

from .models import ForecastRecord
from django.utils.translation import gettext as _

@admin.register(ForecastRecord)
class ForecastRecordAdmin(admin.ModelAdmin):
    """预测记录管理后台配置"""
    list_display = ('forecast_type', 'predicted_value', 'actual_value', 'prediction_date', 'is_alerted')
    list_filter = ('forecast_type', 'is_alerted')
    search_fields = ('prediction_date', 'model_version')
    date_hierarchy = 'prediction_date'
    readonly_fields = ('prediction_date', 'model_version')
    fieldsets = (
        (_('基本信息'), {
            'fields': ('forecast_type', 'predicted_value', 'actual_value', 'prediction_date', 'model_version')
        }),
        (_('预警信息'), {
            'fields': ('is_alerted', 'alert_deviation', 'alert_threshold')
        }),
    )