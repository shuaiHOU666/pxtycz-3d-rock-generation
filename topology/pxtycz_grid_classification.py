"""
Pore Topology Model Usage Example (Grid Classification Version)
Usage:
1. Load saved model (with solidified StandardScaler, PCA weights, and PxTyCz classification boundaries)
2. Classify new data, obtain ternary coordinates and grid-based topology labels
3. Visualize results
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import pickle
import json
from matplotlib.patches import Polygon


# Set font settings (keeping for potential Chinese characters in data)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# Ensure PoreTopologyModel can be imported from your environment
from pxtycz_pca_cluster import *

def load_and_apply_model_with_grid_classification():
    """
    Load model and apply to new data using grid classification
    """
    # 1. Load model
    model_path = os.path.join(os.path.dirname(__file__), 'pore_topology_model.pkl')

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}. Please run the main analysis function to save the model first.")
        return None, None, None, None

    print("=== 1. Loading Model ===")
    model = PoreTopologyModel()
    model.load_model(model_path)
    data_path = './c16/topology.txt'
    # Load data from file
    if os.path.exists(data_path):
        _, _, _, new_data = load_topology_data(data_path)

    print(f"Loaded {len(new_data)} samples")

    # 3. Apply model to get ternary coordinates and basic topology types
    print("\n=== 3. Applying Model for Projection and Classification ===")
    ternary_coords, _ = model.apply_to_new_data(new_data)

    # 4. Perform detailed classification based on grid position
    print("\n=== 4. Performing Detailed Classification Based on Ternary Grid ===")
    grid_categories = classify_by_ternary_grid(ternary_coords)

    # 5. Display results
    print_results(ternary_coords, grid_categories, new_data.index)

    # 6. Visualization (enhanced version, showing grid and classification)
    print("\n=== 6. Visualization ===")
    plot_grid_classification(ternary_coords, grid_categories, new_data.index, model.normalization_params)

    # 7. Save results
    save_results(ternary_coords, grid_categories, new_data.index)

    return ternary_coords, grid_categories, new_data.index

def classify_by_ternary_grid(ternary_coords, n_intervals=4):
    """
    Perform classification based on ternary grid position
    Parameters:
        ternary_coords: Ternary coordinates (N x 3)
        n_intervals: Number of intervals per dimension, default is 4 (P1-P4, T1-T4, C1-C4)
    Returns:
        grid_categories: List of grid classification labels
    """
    print(f"Using {n_intervals}×{n_intervals}×{n_intervals} grid for classification...")

    grid_categories = []

    # For each sample, determine its position in the 3D grid
    for i, coords in enumerate(ternary_coords):
        fc1, fc2, fc3 = coords

        # Divide each dimension into n_intervals
        def get_interval(val, n):
            """Divide 0-1 value into n intervals"""
            if val == 1.0:
                return n
            for j in range(1, n + 1):
                if val <= j / n:
                    return j
            return n

        # Convert ternary coordinates to grid coordinates
        c_interval = get_interval(fc1, n_intervals)  # Connectivity (fc1)
        p_interval = get_interval(fc2, n_intervals)  # Pore size (fc2)
        t_interval = get_interval(fc3, n_intervals)  # Throat size (fc3)


        # Generate grid classification label
        grid_label = f"P{p_interval}T{t_interval}C{c_interval}"
        grid_categories.append(grid_label)

    # Statistics of classification distribution
    unique_categories = set(grid_categories)
    print(f"Generated {len(unique_categories)} different grid classifications")

    # Display first few classifications
    category_counts = {}
    for cat in grid_categories:
        category_counts[cat] = category_counts.get(cat, 0) + 1

    print("Classification Distribution Statistics:")
    for cat, count in sorted(category_counts.items()):
        percentage = count / len(grid_categories) * 100
        print(f"  {cat}: {count} samples ({percentage:.1f}%)")

    return grid_categories

def print_results(ternary_coords, grid_categories, sample_names):
    """
    Print results
    """
    print("\n=== 5. Results Summary ===")
    print("=" * 90)
    print(f"{'Sample':<10} {'FC1':<8} {'FC2':<8} {'FC3':<8} {'Sum':<8} {'PxTyCz':<10} {'Grid Class':<12} {'Region Desc':<20}")
    print("=" * 90)

    for i, name in enumerate(sample_names):
        fc1, fc2, fc3 = ternary_coords[i]
        coord_sum = fc1 + fc2 + fc3
        # base_cat = base_categories[i]
        grid_cat = grid_categories[i]

        # Add description based on grid classification
        region_desc = get_region_description(grid_cat)

        print(f"{name:<10} {fc1:<8.4f} {fc2:<8.4f} {fc3:<8.4f} {coord_sum:<8.4f}{grid_cat:<12} {region_desc:<20}")

    print("=" * 90)

    # Grid classification explanation
    print("\nGrid Classification Explanation:")
    print("P: Pore Size Dimension")
    print("  P1: Small pores (0-25%)  P2: Small-medium pores (25-50%)  P3: Medium-large pores (50-75%)  P4: Large pores (75-100%)")
    print("T: Throat Size Dimension")
    print("  T1: Small throats (0-25%)  T2: Small-medium throats (25-50%)  T3: Medium-large throats (50-75%)  T4: Large throats (75-100%)")
    print("C: Connectivity Dimension")
    print("  C1: Low connectivity (0-25%)  C2: Medium-low connectivity (25-50%)  C3: Medium-high connectivity (50-75%)  C4: High connectivity (75-100%)")


def plot_grid_classification(ternary_coords, grid_categories, sample_names, normalization_params=None):
    """
    Plot ternary diagram with grid - 简化版本，只显示PTC坐标标签
    """
    print("Generating ternary diagram with grid...")

    # Create output directory
    output_dir = './grid_classification_results'
    os.makedirs(output_dir, exist_ok=True)

    # ========== 保持原始代码 ==========
    fc1 = ternary_coords[:, 0]  # Coordinate Number
    fc2 = ternary_coords[:, 1]  # Pore Size
    fc3 = ternary_coords[:, 2]  # Throat Size

    # 坐标转换
    x = fc2 + 0.5 * fc1
    y = (np.sqrt(3) / 2) * fc1

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 11))

    # ===== Subplot 1: Ternary diagram =====
    triangle = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])
    ax1.plot(triangle[:, 0], triangle[:, 1], 'k-', linewidth=3)

    # ========== 绘制网格线 ==========
    n_intervals = 4
    grid_line_style = dict(color='lightblue', alpha=0.5, linewidth=1, linestyle='--')

    for i in range(1, n_intervals):
        frac = i / n_intervals

        # Lines parallel to base (水平线)
        ax1.plot([frac / 2, 1 - frac / 2],
                 [frac * np.sqrt(3) / 2, frac * np.sqrt(3) / 2],
                 **grid_line_style)

        # Lines parallel to right side (右侧平行线)
        ax1.plot([frac, 0.5 + frac / 2],
                 [0, (1 - frac) * np.sqrt(3) / 2],
                 **grid_line_style)

        # Lines parallel to left side (左侧平行线)
        ax1.plot([1 - frac, 0.5 - frac / 2],
                 [0, (1 - frac) * np.sqrt(3) / 2],
                 **grid_line_style)

    # ========== 计算网格区域中心点 ==========
    print("\n=== 计算网格区域 ===")

    # 创建一个字典来存储每个网格区域的样本
    region_samples = {}

    # 统计每个区域的样本
    for i, grid_cat in enumerate(grid_categories):
        if grid_cat not in region_samples:
            region_samples[grid_cat] = []
        region_samples[grid_cat].append((sample_names[i], x[i], y[i], i))

    # 计算每个网格区域的中心点（所有样本点的平均）
    region_centers = {}

    for grid_cat, samples in region_samples.items():
        if samples:
            # 计算该区域所有样本的平均位置
            avg_x = np.mean([s[1] for s in samples])
            avg_y = np.mean([s[2] for s in samples])
            sample_count = len(samples)

            # 解析PTC值
            try:
                p_val = int(grid_cat[1])
                t_val = int(grid_cat[3])
                c_val = int(grid_cat[5])
            except:
                p_val = t_val = c_val = 0

            region_centers[grid_cat] = {
                'x': avg_x,
                'y': avg_y,
                'count': sample_count,
                'p': p_val,
                't': t_val,
                'c': c_val,
                'samples': samples
            }

            print(f"区域 {grid_cat}: {sample_count}样本, 中心位置({avg_x:.3f}, {avg_y:.3f})")

            # 计算所有可能的网格区域标签
            grid_centers = {}

            # 遍历所有可能的P、T、C组合 (1-4)
            for p in range(1, n_intervals + 1):
                for t in range(1, n_intervals + 1):
                    for c in range(1, n_intervals + 1):
                        # 计算网格中心（在重心坐标系中）
                        p_val = (p - 0.5) / n_intervals
                        t_val = (t - 0.5) / n_intervals
                        c_val = (c - 0.5) / n_intervals

                        # 归一化确保和为1
                        total = p_val + t_val + c_val
                        if total > 0:
                            p_val /= total
                            t_val /= total
                            c_val /= total

                        # 转换为直角坐标
                        x_center = p_val + 0.5 * c_val
                        y_center = (np.sqrt(3) / 2) * c_val

                        grid_label = f"P{p}T{t}C{c}"
                        grid_centers[grid_label] = (x_center, y_center, p, t, c)

        # 统计每个网格区域的样本数
        grid_sample_counts = {}
        for cat in grid_categories:
            grid_sample_counts[cat] = grid_sample_counts.get(cat, 0) + 1

        print(f"\nGrid region statistics:")
        print(f"Total possible grid regions: {len(grid_centers)}")
        print(f"Regions with samples: {len(grid_sample_counts)}")

        # 只标注有样本的网格区域（移除无样本区域的灰色标签）
        for grid_label, (x_center, y_center, p, t, c) in grid_centers.items():
            if grid_label in grid_sample_counts and grid_sample_counts[grid_label] > 0:
                # 有样本的区域 - 使用显眼的颜色
                count = grid_sample_counts[grid_label]

                # 统一使用深蓝色标签，白色背景
                color = 'darkblue'
                bg_color = 'white'
                edge_color = 'blue'
                fontsize = 9
                bg_alpha = 0.7

                # # 根据样本数量调整字体大小和背景透明度
                # if count >= 3:
                #     fontsize = 10
                #     bg_alpha = 0.9
                # elif count == 2:
                #     fontsize = 9
                #     bg_alpha = 0.8
                # else:
                #     fontsize = 8
                #

                # 检查点是否在三角形内部
                def point_in_triangle(px, py):
                    A = (0, 0)
                    B = (1, 0)
                    C = (0.5, np.sqrt(3) / 2)

                    def sign(p1, p2, p3):
                        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

                    d1 = sign((px, py), A, B)
                    d2 = sign((px, py), B, C)
                    d3 = sign((px, py), C, A)

                    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
                    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

                    return not (has_neg and has_pos)

                if point_in_triangle(x_center, y_center):
                    ax1.text(x_center, y_center, grid_label,
                             fontsize=fontsize, ha='center', va='center',
                             color=color, fontweight='bold',
                             bbox=dict(boxstyle='round,pad=0.2',
                                       facecolor=bg_color,
                                       alpha=bg_alpha,
                                       edgecolor=edge_color,
                                       linewidth=1.5))

    # ========== 在网格区域中心只显示PTC坐标标签 ==========
    print("\n=== 显示网格区域PTC标签 ===")

    # 收集所有标签位置，用于检测重叠
    label_positions = []

    for grid_cat, region_info in region_centers.items():
        x_center = region_info['x']
        y_center = region_info['y']
        count = region_info['count']

        # 检查标签位置是否与其他标签或样本太接近
        label_too_close = False
        adjusted_position = (x_center, y_center)

        # 检查是否与其他标签重叠
        for pos in label_positions:
            distance = np.sqrt((x_center - pos[0]) ** 2 + (y_center - pos[1]) ** 2)
            if distance < 0.08:  # 标签间最小距离
                label_too_close = True
                # 尝试调整位置
                angle = np.random.uniform(0, 2 * np.pi)
                adjusted_position = (
                    x_center + 0.04 * np.cos(angle),
                    y_center + 0.04 * np.sin(angle)
                )
                print(f"  调整 {grid_cat} 标签位置避免重叠")
                break

        # 检查是否与样本点太接近
        for sample_name, sample_x, sample_y, _ in region_info['samples']:
            distance = np.sqrt((adjusted_position[0] - sample_x) ** 2 +
                               (adjusted_position[1] - sample_y) ** 2)
            if distance < 0.06:  # 标签与样本最小距离
                # 进一步调整
                angle = np.arctan2(adjusted_position[1] - sample_y,
                                   adjusted_position[0] - sample_x)
                adjusted_position = (
                    sample_x + 0.07 * np.cos(angle),
                    sample_y + 0.07 * np.sin(angle)
                )
                print(f"  调整 {grid_cat} 标签位置避免与样本重合")
                break

        final_x, final_y = adjusted_position
        label_positions.append((final_x, final_y))

        # 在网格区域中心添加PTC标签
        label_text = grid_cat  # 只显示PTC坐标，如P1T2C3

        # 添加PTC标签 - 简洁样式
        ax1.text(final_x, final_y, label_text,
                 fontsize=8, ha='center', va='center',
                 color='black', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.1',
                           facecolor='white',
                           alpha=0.85,
                           edgecolor='gray',
                           linewidth=0.8),
                 zorder=10)

        print(f"显示标签: {grid_cat} 在 ({final_x:.3f}, {final_y:.3f})")

    # ========== 简化轴标签（只保留数值） ==========
    tick_fontsize = 8

    # 底部边 - 只保留数值刻度
    for i in range(n_intervals + 1):
        frac = i / n_intervals
        tick_label = f"{frac:.1f}" if frac < 1 else "1.0"

        # 添加刻度标记
        ax1.plot([frac, frac], [-0.01, -0.005], 'k-', linewidth=0.8)
        # 只显示数值标签
        ax1.text(frac, -0.02, tick_label, ha='center', va='top',
                 fontsize=tick_fontsize, color='black')

    # 右侧边 - 只保留数值刻度
    for i in range(n_intervals + 1):
        frac = i / n_intervals
        tick_label = f"{frac:.1f}" if frac < 1 else "1.0"

        x_pos = 1 - frac / 2
        y_pos = frac * np.sqrt(3) / 2

        angle = -60

        dx = 0.01 * np.cos(np.radians(angle))
        dy = 0.01 * np.sin(np.radians(angle))
        ax1.plot([x_pos, x_pos + dx], [y_pos, y_pos + dy], 'k-', linewidth=0.8)

        # 只显示数值标签
        ax1.text(x_pos + 1.5 * dx, y_pos + 1.5 * dy, tick_label,
                 ha='center', va='center', fontsize=tick_fontsize,
                 color='black', rotation=angle)

    # 左侧边 - 只保留数值刻度
    for i in range(n_intervals + 1):
        frac = (i / n_intervals)
        tick_label = f"{frac:.1f}" if frac < 1 else "1.0"
        position_frac = 1 - frac

        x_pos = position_frac / 2
        y_pos = position_frac * np.sqrt(3) / 2

        angle = 60

        dx = 0.01 * np.cos(np.radians(angle))
        dy = 0.01 * np.sin(np.radians(angle))
        ax1.plot([x_pos, x_pos + dx], [y_pos, y_pos + dy], 'k-', linewidth=0.8)

        # 只显示数值标签
        ax1.text(x_pos + 1.5 * dx, y_pos + 1.5 * dy, tick_label,
                 ha='center', va='center', fontsize=tick_fontsize,
                 color='black', rotation=-angle)

    # ========== 可选：简单顶点标记 ==========
    show_simple_vertices = False
    if show_simple_vertices:
        ax1.scatter([0, 1, 0.5], [0, 0, np.sqrt(3) / 2],
                    s=20, color=['blue', 'green', 'red'],
                    alpha=0.5, zorder=1)

    # ========== 绘制样本点 ==========
    unique_categories = sorted(set(grid_categories))

    # 配色方案 - 使用更清晰的配色
    if len(unique_categories) <= 12:
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_categories)))
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_categories)))

    color_map = {}
    for i, cat in enumerate(unique_categories):
        color_map[cat] = colors[i]

    print("\n=== 样本信息 ===")
    for i, (xi, yi, cat) in enumerate(zip(x, y, grid_categories)):
        color = color_map[cat]

        # 绘制样本点
        ax1.scatter(xi, yi, s=100, color=color, alpha=0.9,
                    edgecolors='black', linewidth=1.0, zorder=5)

        # 显示样本名
        ax1.text(xi, yi + 0.015, sample_names[i], fontsize=7, ha='center', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.1',
                           facecolor='white',
                           alpha=0.9,
                           edgecolor=color,
                           linewidth=0.8))

        print(f"样本 {sample_names[i]}: 位置({xi:.3f}, {yi:.3f}), 区域 {cat}")

    ax1.set_aspect('equal')
    ax1.set_xlim(-0.08, 1.08)
    ax1.set_ylim(-0.08, 0.95)
    ax1.axis('off')
    ax1.set_title('Pore Topology Classification\nGrid Regions: PxTyCz',
                  fontsize=14, fontweight='bold', pad=20)

    # ========== 添加图例 ==========
    if len(unique_categories) <= 20:
        legend_elements = []
        for cat in unique_categories:
            color = color_map[cat]
            count = len([c for c in grid_categories if c == cat])
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                              markerfacecolor=color, markersize=6,
                                              label=f'{cat} ({count})'))

        ncol = 2 if len(unique_categories) <= 10 else 3
        ax1.legend(handles=legend_elements, loc='upper center',
                   bbox_to_anchor=(0.5, -0.15), ncol=ncol, fontsize=6,
                   title=f"Sample Classifications",
                   title_fontsize=7)

        # ===== Subplot 2: 3D coordinate distribution =====
        ax2 = fig.add_subplot(122, projection='3d')

        # 为3D图使用相同颜色方案
        for i, cat in enumerate(unique_categories):
            mask = [c == cat for c in grid_categories]
            if any(mask):
                color = color_map[cat]
                ax2.scatter(fc1[mask], fc2[mask], fc3[mask],
                            color=color, s=80, alpha=0.8,
                            edgecolor='black', linewidth=0.5, label=cat)

        # 设置坐标轴标签
        ax2.set_xlabel('Coordinate Number\n(Topological Connectivity)', fontsize=11, labelpad=10)
        ax2.set_ylabel('Pore Size\n(Storage Scale)', fontsize=11, labelpad=10)
        ax2.set_zlabel('Throat Size\n(Transport Constraint)', fontsize=11, labelpad=10)
        ax2.set_title('3D Coordinate Distribution', fontsize=14, fontweight='bold', pad=20)

        # 设置坐标轴范围
        ax2.set_xlim([0, 1])
        ax2.set_ylim([0, 1])
        ax2.set_zlim([0, 1])

        # 添加网格和调整视角
        ax2.grid(True, alpha=0.3)
        ax2.view_init(elev=25, azim=45)

        # 为所有样本添加标签
        for i, (x_val, y_val, z_val, name, cat) in enumerate(zip(fc1, fc2, fc3, sample_names, grid_categories)):
            ax2.text(x_val, y_val, z_val, f'{name}',
                     fontsize=7, ha='center', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.7))

        plt.tight_layout()

        # 保存图像
        plot_file = os.path.join(output_dir, 'ternary_grid_classification.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"Grid classification plot saved to: {plot_file}")

        # 打印详细的网格统计信息
        print("\n" + "=" * 60)
        print("Grid Regions with Samples:")
        print("=" * 60)

        # 按样本数排序
        sorted_regions = sorted(grid_sample_counts.items(), key=lambda x: x[1], reverse=True)

        for cat, count in sorted_regions:
            region_desc = get_region_description(cat)
            print(f"{cat}: {count} sample(s) - {region_desc}")

        print("=" * 60)
    return fig




def create_simplified_plot(ternary_coords, grid_categories, sample_names, output_dir):
    """创建简化的分类图，突出显示分类结果"""
    fig = plt.figure(figsize=(12, 10))
    ax = plt.gca()

    # 坐标转换
    fc1 = ternary_coords[:, 0]
    fc2 = ternary_coords[:, 1]
    fc3 = ternary_coords[:, 2]

    x = fc2 + 0.5 * fc1
    y = (np.sqrt(3) / 2) * fc1

    # 绘制三角形
    triangle = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])
    ax.plot(triangle[:, 0], triangle[:, 1], 'k-', linewidth=2)

    # 设置顶点标签
    ax.text(0, -0.05, 'Pore Size', ha='center', va='top', fontsize=11, fontweight='bold')
    ax.text(1, -0.05, 'Throat Size', ha='center', va='top', fontsize=11, fontweight='bold')
    ax.text(0.5, np.sqrt(3) / 2 + 0.05, 'Coord Number', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 按分类着色
    unique_categories = sorted(set(grid_categories))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_categories)))
    color_map = {cat: colors[i] for i, cat in enumerate(unique_categories)}

    # 绘制样本点
    for i, (xi, yi, cat) in enumerate(zip(x, y, grid_categories)):
        color = color_map[cat]
        ax.scatter(xi, yi, s=100, color=color, alpha=0.8,
                   edgecolors='black', linewidth=1, zorder=5)
        ax.text(xi, yi + 0.015, sample_names[i], fontsize=8, ha='center', va='bottom')

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Pore Topology Classification Results', fontsize=14, fontweight='bold', pad=20)

    # 添加图例
    legend_elements = []
    for cat in unique_categories[:min(8, len(unique_categories))]:
        color = color_map[cat]
        desc = get_region_description(cat)
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor=color, markersize=8,
                                          label=f'{cat}: {desc}'))

    if legend_elements:
        ax.legend(handles=legend_elements, loc='center left',
                  bbox_to_anchor=(1.05, 0.5), fontsize=9)

    plt.tight_layout()
    plt.show()
    # simple_file = os.path.join(output_dir, 'simplified_classification.png')
    # plt.savefig(simple_file, dpi=300, bbox_inches='tight')
    # plt.close()
    # print(f"Simplified plot saved to: {simple_file}")


# 更新顶点标签描述函数
def get_region_description(grid_category):
    """
    Get region description based on grid classification
    """
    # Parse classification label
    p_val = int(grid_category[1])
    t_val = int(grid_category[3])
    c_val = int(grid_category[5])

    # Determine dominant dimension
    max_val = max(p_val, t_val, c_val)

    if max_val == p_val and p_val > t_val and p_val > c_val:
        return "Pore-dominant"
    elif max_val == t_val and t_val > p_val and t_val > c_val:
        return "Throat-dominant"
    elif max_val == c_val and c_val > p_val and c_val > t_val:
        return "Connectivity-dominant"
    elif p_val == t_val == c_val:
        return "Balanced"
    elif p_val == t_val and p_val > c_val:
        return "Pore-Throat balanced"
    elif p_val == c_val and p_val > t_val:
        return "Pore-Connectivity balanced"
    elif t_val == c_val and t_val > p_val:
        return "Throat-Connectivity balanced"
    else:
        return "Mixed type"

def save_results(ternary_coords, grid_categories, sample_names):
    """
    Save results to files
    """
    output_dir = './c16_lr_grid_classification_results'
    os.makedirs(output_dir, exist_ok=True)

    # Create results DataFrame
    results_df = pd.DataFrame({
        'sample_name': sample_names,
        'fc1_connectivity': ternary_coords[:, 0],
        'fc2_pore_size': ternary_coords[:, 1],
        'fc3_throat_size': ternary_coords[:, 2],
        'coordinate_sum': np.sum(ternary_coords, axis=1),
        'grid_classification': grid_categories,
        'region_description': [get_region_description(cat) for cat in grid_categories]
    })

    # 1. Save as CSV
    csv_file = os.path.join(output_dir, 'grid_classification_results.csv')
    results_df.to_csv(csv_file, index=False, encoding='utf-8')
    print(f"CSV results saved to: {csv_file}")

    print(f"\nAll results saved to directory: {output_dir}")

if __name__ == "__main__":
    print("Pore Topology Grid Classification System")
    print("=" * 60)
    print("Classification method based on ternary diagram grid positions (PxTyCz format)")
    print("=" * 60)

    try:
        # Run grid classification system
        ternary_coords, grid_categories, sample_names = load_and_apply_model_with_grid_classification()

        if ternary_coords is not None:
            print("\n" + "=" * 60)
            print("Processing completed!")
            print("=" * 60)
            print(f"Processed {len(sample_names)} samples")
            print(f"Generated {len(set(grid_categories))} grid classifications")
            print(f"Covering {len(set([get_region_description(cat) for cat in grid_categories]))} region types")
            print("Results saved to ./grid_classification_results/ directory")

            # Show classification examples
            print("\nClassification Examples:")
            for i in range(min(3, len(sample_names))):
                print(f"  {sample_names[i]}: {grid_categories[i]} ({get_region_description(grid_categories[i])})")
    except Exception as e:
        print(f"\nExecution failed: {e}")
        import traceback
        traceback.print_exc()