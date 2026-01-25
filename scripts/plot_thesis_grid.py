import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_thesis_results():
    # 1. 定义文件路径
    results_path = r"E:\graduation_project\Mask-Wearing-Detection\runs\detect\runs\mask_detection_thesis\yolov11_200epochs\results.csv"
    output_path = r"E:\graduation_project\Mask-Wearing-Detection\runs\detect\runs\mask_detection_thesis\yolov11_200epochs\thesis_custom_plot_f1.png"

    if not os.path.exists(results_path):
        print(f"错误: 找不到文件 {results_path}")
        return

    # 2. 读取数据
    try:
        df = pd.read_csv(results_path)
    except Exception as e:
        print(f"读取CSV出错: {e}")
        return

    # 清理列名（去除首尾空格）
    df.columns = df.columns.str.strip()
    
    # 3. 计算 F1 Score
    # F1 = 2 * (Precision * Recall) / (Precision + Recall)
    # 加上一个小数值防止除以零
    eps = 1e-16
    if 'metrics/precision(B)' in df.columns and 'metrics/recall(B)' in df.columns:
        df['f1'] = 2 * (df['metrics/precision(B)'] * df['metrics/recall(B)']) / (df['metrics/precision(B)'] + df['metrics/recall(B)'] + eps)
    else:
        print("警告: 缺少 Precision 或 Recall 列，无法计算 F1")
        df['f1'] = 0

    # 4. 创建画布 (4行2列)
    fig, axes = plt.subplots(4, 2, figsize=(12, 16), dpi=300)
    
    # 设置全局样式微调
    plt.rcParams['axes.grid'] = True
    plt.rcParams['font.family'] = 'sans-serif' # 避免中文字体问题，使用通用无衬线

    # 定义绘图数据映射 (Title, Column Name, YLabel)
    plots_config = [
        ("Box Loss", "train/box_loss", "train/box_loss"),
        ("Cls Loss", "train/cls_loss", "train/cls_loss"),
        ("DFL Loss", "train/dfl_loss", "train/dfl_loss"),
        ("mAP@0.5", "metrics/mAP50(B)", "metrics/mAP50(B)"),
        ("mAP@0.5:0.95", "metrics/mAP50-95(B)", "metrics/mAP50-95(B)"),
        ("Precision", "metrics/precision(B)", "metrics/precision(B)"),
        ("Recall", "metrics/recall(B)", "metrics/recall(B)"),
        ("F1 Score", "f1", "F1 Score")  # 第8张图
    ]

    # 5. 循环绘图
    flat_axes = axes.flatten()
    epochs = df['epoch']

    for i, (ax, (title, col, ylabel)) in enumerate(zip(flat_axes, plots_config)):
        if col in df.columns:
            ax.plot(epochs, df[col], linewidth=1.5, color='#2874A6') # 使用类似的深蓝色
            ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.grid(True, linestyle=':', alpha=0.6) # 点状网格
            
            # 优化坐标轴边距
            ax.margins(x=0.01)
        else:
            ax.text(0.5, 0.5, f'Missing Data:\n{col}', ha='center', va='center')

    # 6. 调整布局
    plt.tight_layout()
    
    # 7. 保存文件
    plt.savefig(output_path, bbox_inches='tight')
    print(f"绘图完成！图片已保存至: {output_path}")

if __name__ == "__main__":
    plot_thesis_results()
