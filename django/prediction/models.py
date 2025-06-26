from django.db import models

# Create your models here.
# prediction/models.py
from django.utils.translation import gettext_lazy as _

class ForecastRecord(models.Model):
    """预测记录模型，用于存储预测结果和预警状态"""
    
    # 预测类型选择（周度/月度）
    WEEKLY = 'weekly'
    MONTHLY = 'monthly'
    FORECAST_TYPES = (
        (WEEKLY, _('周度预测')),
        (MONTHLY, _('月度预测')),
    )
    
    # 基础字段
    forecast_type = models.CharField(
        _('预测类型'),
        max_length=20,
        choices=FORECAST_TYPES,
        default=WEEKLY
    )
    predicted_value = models.FloatField(_('预测值'))
    actual_value = models.FloatField(
        _('实际值'), 
        null=True, 
        blank=True,
        help_text=_('后续录入的实际收入值')
    )
    prediction_date = models.DateTimeField(
        _('预测日期'), 
        auto_now_add=True
    )
    model_version = models.CharField(
        _('模型版本'), 
        max_length=50,
        default='v1.0'
    )
    
    # 预警相关字段
    is_alerted = models.BooleanField(
        _('是否触发预警'), 
        default=False
    )
    alert_deviation = models.FloatField(
        _('预警偏差'), 
        null=True, 
        blank=True,
        help_text=_('触发预警时的偏差百分比')
    )
    alert_threshold = models.FloatField(
        _('预警阈值'), 
        default=0.1,
        help_text=_('偏差超过此阈值时触发预警')
    )
    
    # 元数据
    class Meta:
        verbose_name = _('预测记录')
        verbose_name_plural = _('预测记录列表')
        ordering = ['-prediction_date']  # 按预测日期降序排列
    
    # 对象字符串表示
    def __str__(self):
        return f"{self.get_forecast_type_display()} - {self.prediction_date.strftime('%Y-%m-%d')}"
    
    # 计算偏差（用于实际值录入后）
    def calculate_deviation(self):
        if self.actual_value and self.predicted_value:
            return abs(self.predicted_value - self.actual_value) / self.actual_value
        return 0