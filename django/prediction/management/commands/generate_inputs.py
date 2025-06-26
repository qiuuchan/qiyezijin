# prediction/management/commands/generate_inputs.py
from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = '生成所有预测输入数据'
    
    def handle(self, *args, **options):
        self.stdout.write("开始生成周度输入数据...")
        from .generate_weekly_input import Command as WeeklyCommand
        WeeklyCommand().handle()
        
        self.stdout.write("开始生成月度输入数据...")
        from .generate_monthly_input import Command as MonthlyCommand
        MonthlyCommand().handle()
        
        self.stdout.write(self.style.SUCCESS('所有预测输入数据生成完成'))