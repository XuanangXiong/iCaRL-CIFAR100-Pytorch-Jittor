#!/usr/bin/env python3
"""
PyTorch vs Jittor iCaRL 日志对比可视化脚本
读取两个框架的JSON日志文件，生成对比可视化图表
"""

import json
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class LogComparator:
    def __init__(self, torch_results_dir="Pytorch_results", jittor_results_dir="Jittor_results"):
        self.torch_dir = torch_results_dir
        self.jittor_dir = jittor_results_dir
        self.torch_data = {}
        self.jittor_data = {}

    def load_data(self):
        """加载两个框架的日志数据"""
        print("加载PyTorch日志...")
        self.torch_data = self._load_framework_data(self.torch_dir, "PyTorch")

        print("加载Jittor日志...")
        self.jittor_data = self._load_framework_data(self.jittor_dir, "Jittor")

        if not self.torch_data and not self.jittor_data:
            print("警告：未找到任何日志文件!")
            return False
        return True

    def _load_framework_data(self, results_dir, framework_name):
        """加载指定框架的数据"""
        data = {}
        if not os.path.exists(results_dir):
            print(f"  {framework_name} 结果目录不存在: {results_dir}")
            return data

        json_files = glob.glob(os.path.join(results_dir, "results_task_*.json"))

        for json_file in sorted(json_files):
            try:
                with open(json_file, 'r') as f:
                    task_data = json.load(f)
                    task_id = task_data.get('current_task', 0)
                    data[task_id] = task_data
                    print(f"  加载 {framework_name} Task {task_id}")
            except Exception as e:
                print(f"  加载失败 {json_file}: {e}")

        return data

    def plot_comparison(self):
        """生成对比图表"""
        if not self.torch_data and not self.jittor_data:
            print("没有数据可供可视化")
            return

        # 创建输出目录
        output_dir = "comparison_results"
        os.makedirs(output_dir, exist_ok=True)

        # 1. 准确率对比
        self._plot_accuracy_comparison(output_dir)

        # 2. Loss曲线对比
        self._plot_loss_comparison(output_dir)

        # 3. 遗忘对比
        self._plot_forgetting_comparison(output_dir)

        # 4. 综合对比表格
        self._plot_summary_table(output_dir)

        print(f"\n对比结果已保存到 {output_dir}/ 目录")

    def _plot_accuracy_comparison(self, output_dir):
        """绘制准确率对比"""
        plt.figure(figsize=(15, 10))

        # 提取数据
        torch_current = self._extract_metric(self.torch_data, 'fc_current_accuracies')
        torch_cumulative = self._extract_metric(self.torch_data, 'fc_cumulative_accuracies')
        jittor_current = self._extract_metric(self.jittor_data, 'fc_current_accuracies')
        jittor_cumulative = self._extract_metric(self.jittor_data, 'fc_cumulative_accuracies')

        # 子图1: 当前任务准确率
        plt.subplot(2, 2, 1)
        if torch_current: plt.plot(torch_current, 'b-o', label='PyTorch', linewidth=2)
        if jittor_current: plt.plot(jittor_current, 'r-s', label='Jittor', linewidth=2)
        plt.xlabel('Task ID')
        plt.ylabel('Accuracy (%)')
        plt.title('Current Task Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 子图2: 累计准确率
        plt.subplot(2, 2, 2)
        if torch_cumulative: plt.plot(torch_cumulative, 'b-o', label='PyTorch', linewidth=2)
        if jittor_cumulative: plt.plot(jittor_cumulative, 'r-s', label='Jittor', linewidth=2)
        plt.xlabel('Task ID')
        plt.ylabel('Accuracy (%)')
        plt.title('Cumulative Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 子图3: NMS对比
        plt.subplot(2, 2, 3)
        torch_nms = self._extract_metric(self.torch_data, 'nms_current_accuracies')
        jittor_nms = self._extract_metric(self.jittor_data, 'nms_current_accuracies')
        if torch_nms: plt.plot(torch_nms, 'b-o', label='PyTorch NMS', linewidth=2)
        if jittor_nms: plt.plot(jittor_nms, 'r-s', label='Jittor NMS', linewidth=2)
        plt.xlabel('Task ID')
        plt.ylabel('Accuracy (%)')
        plt.title('NMS Classifier Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 子图4: 差异分析
        plt.subplot(2, 2, 4)
        if torch_current and jittor_current:
            diff = np.array(jittor_current) - np.array(torch_current)
            plt.bar(range(len(diff)), diff, alpha=0.7,
                    color=['g' if d >= 0 else 'r' for d in diff])
            plt.xlabel('Task ID')
            plt.ylabel('Accuracy Difference (%)')
            plt.title('Jittor - PyTorch (Current Task)')
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.5)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/accuracy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ 准确率对比图已生成")

    def _plot_loss_comparison(self, output_dir):
        """绘制Loss曲线对比"""
        torch_loss = self._extract_metric(self.torch_data, 'loss_history')
        jittor_loss = self._extract_metric(self.jittor_data, 'loss_history')

        if not torch_loss and not jittor_loss:
            print("⚠ 未找到loss数据，跳过loss对比图")
            return

        plt.figure(figsize=(15, 8))

        # 计算最大任务数
        max_tasks = max(len(torch_loss) if torch_loss else 0,
                        len(jittor_loss) if jittor_loss else 0)

        if max_tasks == 0:
            return

        # 为每个任务绘制loss曲线
        cols = min(5, max_tasks)
        rows = (max_tasks + cols - 1) // cols

        for task_id in range(max_tasks):
            plt.subplot(rows, cols, task_id + 1)

            if torch_loss and task_id < len(torch_loss):
                epochs = range(1, len(torch_loss[task_id]) + 1)
                plt.plot(epochs, torch_loss[task_id], 'b-', label='PyTorch', linewidth=2)

            if jittor_loss and task_id < len(jittor_loss):
                epochs = range(1, len(jittor_loss[task_id]) + 1)
                plt.plot(epochs, jittor_loss[task_id], 'r-', label='Jittor', linewidth=2)

            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Task {task_id}')
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/loss_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Loss对比图已生成")

    def _plot_forgetting_comparison(self, output_dir):
        """绘制遗忘对比"""
        torch_forgetting = self._extract_metric(self.torch_data, 'forgetting_matrix')
        jittor_forgetting = self._extract_metric(self.jittor_data, 'forgetting_matrix')

        if not torch_forgetting and not jittor_forgetting:
            print("⚠ 未找到遗忘数据，跳过遗忘对比图")
            return

        plt.figure(figsize=(15, 5))

        # 计算平均遗忘率 - 修复版本
        def calc_avg_forgetting(forgetting_data):
            if not forgetting_data:
                return []

            print(f"Debug: forgetting_data keys: {list(forgetting_data.keys())}")
            print(f"Debug: forgetting_data structure: {type(forgetting_data)}")

            avg_forgetting = []

            # 遍历每个任务
            for task_key in sorted(forgetting_data.keys(), key=lambda x: int(x) if isinstance(x, str) else x):
                task_id = int(task_key) if isinstance(task_key, str) else task_key
                task_data = forgetting_data[task_key]

                print(f"Debug: Task {task_id} data: {task_data}")

                if isinstance(task_data, dict):
                    # 如果是字典，计算该任务的平均遗忘
                    task_forgetting_values = []
                    for time_key, forgetting_value in task_data.items():
                        time_point = int(time_key) if isinstance(time_key, str) else time_key
                        if time_point > task_id:  # 只考虑任务学完后的遗忘
                            if isinstance(forgetting_value, (int, float)):
                                task_forgetting_values.append(forgetting_value)

                    if task_forgetting_values:
                        avg_forgetting.append(np.mean(task_forgetting_values))
                    else:
                        avg_forgetting.append(0)
                else:
                    # 如果直接是数值，说明数据结构不同
                    avg_forgetting.append(0)

            return avg_forgetting

        try:
            torch_avg_forgetting = calc_avg_forgetting(torch_forgetting)
            jittor_avg_forgetting = calc_avg_forgetting(jittor_forgetting)
        except Exception as e:
            print(f"计算遗忘率时出错: {e}")
            print("跳过遗忘对比图生成")
            plt.close()
            return

        plt.subplot(1, 2, 1)
        if torch_avg_forgetting:
            x_pos = range(len(torch_avg_forgetting))
            plt.bar([x - 0.2 for x in x_pos], torch_avg_forgetting,
                    width=0.4, alpha=0.7, label='PyTorch', color='blue')
        if jittor_avg_forgetting:
            x_pos = range(len(jittor_avg_forgetting))
            plt.bar([x + 0.2 for x in x_pos], jittor_avg_forgetting,
                    width=0.4, alpha=0.7, label='Jittor', color='red')

        plt.xlabel('Task ID')
        plt.ylabel('Average Forgetting (%)')
        plt.title('Average Forgetting by Task')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 总体遗忘比较
        plt.subplot(1, 2, 2)
        frameworks = []
        avg_values = []

        if torch_avg_forgetting:
            positive_values = [f for f in torch_avg_forgetting if f > 0]
            if positive_values:
                frameworks.append('PyTorch')
                avg_values.append(np.mean(positive_values))

        if jittor_avg_forgetting:
            positive_values = [f for f in jittor_avg_forgetting if f > 0]
            if positive_values:
                frameworks.append('Jittor')
                avg_values.append(np.mean(positive_values))

        if frameworks:
            colors = ['blue', 'red'][:len(frameworks)]
            plt.bar(frameworks, avg_values, color=colors, alpha=0.7)
            plt.ylabel('Overall Average Forgetting (%)')
            plt.title('Overall Forgetting Comparison')
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'No significant forgetting detected',
                     ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Overall Forgetting Comparison')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/forgetting_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ 遗忘对比图已生成")

    def _plot_summary_table(self, output_dir):
        """生成综合对比表格"""
        plt.figure(figsize=(12, 8))
        plt.axis('tight')
        plt.axis('off')

        # 准备表格数据
        headers = ['Metric', 'PyTorch', 'Jittor', 'Difference']
        rows = []

        # 最终累计准确率
        torch_final_acc = self._get_final_metric(self.torch_data, 'fc_cumulative_accuracies')
        jittor_final_acc = self._get_final_metric(self.jittor_data, 'fc_cumulative_accuracies')
        if torch_final_acc is not None and jittor_final_acc is not None:
            diff = jittor_final_acc - torch_final_acc
            rows.append(['Final Cumulative Accuracy (%)',
                         f'{torch_final_acc:.2f}', f'{jittor_final_acc:.2f}', f'{diff:+.2f}'])

        # 平均当前任务准确率
        torch_avg_current = self._get_avg_metric(self.torch_data, 'fc_current_accuracies')
        jittor_avg_current = self._get_avg_metric(self.jittor_data, 'fc_current_accuracies')
        if torch_avg_current is not None and jittor_avg_current is not None:
            diff = jittor_avg_current - torch_avg_current
            rows.append(['Avg Current Task Accuracy (%)',
                         f'{torch_avg_current:.2f}', f'{jittor_avg_current:.2f}', f'{diff:+.2f}'])

        # 总训练任务数
        torch_tasks = len(self.torch_data)
        jittor_tasks = len(self.jittor_data)
        rows.append(['Total Tasks Completed', str(torch_tasks), str(jittor_tasks),
                     str(jittor_tasks - torch_tasks)])

        # 框架信息
        torch_config = self._get_config_info(self.torch_data)
        jittor_config = self._get_config_info(self.jittor_data)

        if torch_config.get('epochs') and jittor_config.get('epochs'):
            rows.append(['Epochs per Task', str(torch_config['epochs']),
                         str(jittor_config['epochs']), 'Same'])

        # 创建表格
        table = plt.table(cellText=rows, colLabels=headers,
                          cellLoc='center', loc='center',
                          colWidths=[0.3, 0.2, 0.2, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.5, 2)

        # 设置表格样式
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        plt.title('PyTorch vs Jittor iCaRL Performance Summary',
                  fontsize=16, fontweight='bold', pad=20)

        plt.savefig(f'{output_dir}/summary_table.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ 综合对比表格已生成")

    def _extract_metric(self, data, metric_name):
        """从数据中提取指定指标"""
        if not data:
            return []

        # 获取最后一个任务的数据，处理字符串和整数键
        if isinstance(list(data.keys())[0], str):
            last_task = max(data.keys(), key=int)
        else:
            last_task = max(data.keys())
        return data[last_task].get(metric_name, [])

    def _get_final_metric(self, data, metric_name):
        """获取最终指标值"""
        values = self._extract_metric(data, metric_name)
        return values[-1] if values else None

    def _get_avg_metric(self, data, metric_name):
        """获取平均指标值"""
        values = self._extract_metric(data, metric_name)
        return np.mean(values) if values else None

    def _get_config_info(self, data):
        """获取配置信息"""
        if not data:
            return {}
        # 处理字符串和整数键
        if isinstance(list(data.keys())[0], str):
            last_task = max(data.keys(), key=int)
        else:
            last_task = max(data.keys())
        return data[last_task].get('hyperparameters', {})


def main():
    print("=" * 60)
    print("PyTorch vs Jittor iCaRL 日志对比可视化工具")
    print("=" * 60)

    # 检查命令行参数，默认使用当前目录下的结果文件夹
    import sys
    torch_dir = sys.argv[1] if len(sys.argv) > 1 else "Pytorch_results"
    jittor_dir = sys.argv[2] if len(sys.argv) > 2 else "Jittor_results"

    print(f"PyTorch 结果目录: {torch_dir}")
    print(f"Jittor 结果目录: {jittor_dir}")

    # 创建比较器
    comparator = LogComparator(torch_dir, jittor_dir)

    # 加载数据
    if not comparator.load_data():
        print("无法加载数据，退出程序")
        return

    # 生成对比图
    print("\n生成对比可视化...")
    comparator.plot_comparison()

    print("\n" + "=" * 60)
    print("对比完成！查看 comparison_results/ 目录下的结果：")
    print("- accuracy_comparison.png: 准确率对比")
    print("- loss_comparison.png: 损失函数对比")
    print("- forgetting_comparison.png: 遗忘率对比")
    print("- summary_table.png: 综合对比表格")
    print("=" * 60)


if __name__ == "__main__":
    main()