import os
import numpy as np
import pandas as pd
import porespy as ps
import openpnm as op
from skimage.util import view_as_windows
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re
import warnings

warnings.filterwarnings("ignore")

# 设置中文字体为微软雅黑
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def sliding_window_subcubes(voxel, subcube_size=32, stride=32):
    """滑动窗口生成子立方体"""
    return view_as_windows(voxel, (subcube_size, subcube_size, subcube_size), stride)

def classify_throats_by_size(net, resolution_um=4.0):
    """
    Classify throats based on equivalent diameter (in pixels) and a given resolution.

    Classification criteria (after conversion to microns):
    - Macro-throat: diameter > 7.5 um
    - Meso-throat: 0.62 um <= diameter <= 7.5 um
    - Micro-throat: 0.063 um <= diameter < 0.62 um
    - Nano-throat: diameter < 0.063 um

    Parameters:
    -----------
    net : PoreSpy network object
        The extracted pore network.
    resolution_um : float
        The image resolution in microns per pixel.

    Returns:
    --------
    throat_size_labels : list of str
        A list of size classification labels for each throat.
    throat_diameters_um : numpy.ndarray
        The equivalent diameters of throats in microns.
    """

    # Extract equivalent diameter in pixels
    throat_diameters_pixels = net.network['throat.equivalent_diameter']

    # Convert to microns
    throat_diameters_um = throat_diameters_pixels * resolution_um

    # Initialize an array to store size labels
    throat_size_labels = []

    # Classify each throat
    for diam in throat_diameters_um:
        if diam < 0.063:
            throat_size_labels.append('nano_throat')
        elif 0.063 <= diam < 0.62:
            throat_size_labels.append('micro_throat')
        elif 0.62 <= diam <= 7.5:
            throat_size_labels.append('meso_throat')
        else:  # diam > 7.5
            throat_size_labels.append('macro_throat')

    return throat_size_labels, throat_diameters_um

def classify_pores_by_size(net, resolution_um=4.0):
    """
    Classify pores based on equivalent diameter (in pixels) and a given resolution.

    Classification criteria (after conversion to microns):
    - Macropore: diameter > 60 um
    - Mesopore: 30 um <= diameter <= 60 um
    - Micropore: 10 um <= diameter < 30 um
    - Nanopore: diameter < 10 um

    Parameters:
    -----------
    net : PoreSpy network object
        The extracted pore network.
    resolution_um : float
        The image resolution in microns per pixel.

    Returns:
    --------
    size_labels : list of str
        A list of size classification labels for each pore.
    pore_diameters_um : numpy.ndarray
        The equivalent diameters of pores in microns.
    """

    # Extract equivalent diameter in pixels
    pore_diameters_pixels = net.network['pore.equivalent_diameter']

    # Convert to microns
    pore_diameters_um = pore_diameters_pixels * resolution_um

    # Initialize an array to store size labels
    size_labels = []

    # Classify each pore
    for diam in pore_diameters_um:
        if diam < 10:
            size_labels.append('nanopore')
        elif 10 <= diam < 30:
            size_labels.append('micropore')
        elif 30 <= diam <= 60:
            size_labels.append('mesopore')
        else:  # diam > 60
            size_labels.append('macropore')

    return size_labels, pore_diameters_um

def classify_pores_by_connectivity_simple(net):
    """
    Pore classification based on connectivity
    Classification criteria:
    - isolated: coordination number = 0
    - dead_end: coordination number = 1
    - well_connected: coordination number 2-4
    - highly_connected: coordination number > 4
    """
    coords = net.network['pore.coords']
    conns = net.network['throat.conns']

    # Calculate coordination number
    coordination = np.zeros(len(coords))
    for conn in conns:
        coordination[conn[0]] += 1
        coordination[conn[1]] += 1

    topology_types = []

    for i, cn in enumerate(coordination):
        if cn == 0:
            topology_types.append('isolated')
        elif cn == 1:
            topology_types.append('dead_end')
        elif 2 <= cn <= 4:
            topology_types.append('well_connected')
        else:  # cn > 4
            topology_types.append('highly_connected')

    return topology_types


def analyze_simple_connectivity(net):
    """
    Simplified connectivity analysis
    """
    coords = net.network['pore.coords']
    conns = net.network['throat.conns']

    # Calculate coordination number
    coordination = np.zeros(len(coords))
    for conn in conns:
        coordination[conn[0]] += 1
        coordination[conn[1]] += 1

    topology_types = classify_pores_by_connectivity_simple(net)

    print("=== Simplified Connectivity Analysis ===")
    print(f"Total pores: {len(coords)}")
    print(f"Total throats: {len(conns)}")
    print(f"Average coordination number: {np.mean(coordination):.2f}")
    print()

    # Classification statistics
    categories = ['isolated', 'dead_end', 'well_connected', 'highly_connected']
    per = {}  # Initialize dictionary to store proportions
    for category in categories:
        count = topology_types.count(category)
        percentage = count / len(topology_types) * 100
        if count > 0:
            # Calculate average coordination number for this category
            mask = np.array([t == category for t in topology_types])
            avg_coord = np.mean(coordination[mask])
            print(f"{category:15s}: {count:3d} pores ({percentage:5.1f}%), avg coordination: {avg_coord:.1f}")
            # Store proportion in dictionary
            per[category] = percentage

    return coordination, topology_types, per


def plot_comparison(cube, net, path, line_scale=1.0, throat_color='#2c3e50'):
    """
    Visualization based on simplified classification
    """
    fig = plt.figure(figsize=(20, 8))
    ax1 = fig.add_subplot(121, projection='3d')

    if net is not None and 'pore.coords' in net.network:
        coords = net.network['pore.coords']
        conns = net.network['throat.conns']

        # Calculate pore radii
        if 'pore.inscribed_diameter' in net.network:
            pore_diameters = net.network['pore.inscribed_diameter']
            pore_radii = pore_diameters / 2
        elif 'pore.volume' in net.network:
            volumes = net.network['pore.volume']
            pore_radii = (3 * volumes / (4 * np.pi)) ** (1 / 3)
        else:
            pore_radii = np.ones(len(coords)) * 0.5

        # Use simplified classification
        coordination, topology_types, per = analyze_simple_connectivity(net)

        # Simplified color mapping
        connectivity_colors = {
            'isolated': '#e74c3c',  # Red - isolated
            'dead_end': '#f39c12',  # Orange - dead_end
            'well_connected': '#3498db',  # Blue - well_connected
            'highly_connected': '#9b59b6'  # Purple - highly_connected
        }

        # Draw throats
        if len(conns) > 0:
            if 'throat.inscribed_diameter' in net.network:
                throat_diameters = net.network['throat.inscribed_diameter']
                throat_linewidths = np.clip(throat_diameters * line_scale * 5, 1, 8)
            else:
                throat_linewidths = np.ones(len(conns)) * 2

            for i, conn in enumerate(conns):
                if conn[0] >= len(coords) or conn[1] >= len(coords):
                    continue

                start = coords[conn[0]]
                end = coords[conn[1]]

                linewidth = throat_linewidths[i] if i < len(throat_linewidths) else 2

                ax1.plot([start[0], end[0]],
                         [start[1], end[1]],
                         [start[2], end[2]],
                         color=throat_color,
                         linewidth=linewidth,
                         alpha=0.6)

        # Draw pore spheres
        u = np.linspace(0, 2 * np.pi, 12)
        v = np.linspace(0, np.pi, 8)

        for i, (center, radius) in enumerate(zip(coords, pore_radii)):
            if radius > 0 and i < len(topology_types):
                topology = topology_types[i]
                color = connectivity_colors.get(topology, '#95a5a6')

                x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
                y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
                z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))

                ax1.plot_surface(x, y, z, color=color, alpha=0.9, linewidth=0)

        # Simplified legend
        legend_elements = []
        for topology, color in connectivity_colors.items():
            if topology in topology_types:
                legend_elements.append(
                    plt.Line2D([0], [0], marker='o', color='w',
                               markerfacecolor=color, markersize=10,
                               label=topology)
                )

        ax1.legend(handles=legend_elements, loc='upper right', fontsize=10)

        # Title
        avg_coord = np.mean(coordination)
        ax1.set_title(f'Pore Network Connectivity Classification\n(Average Coordination: {avg_coord:.2f})', fontsize=12,
                      fontweight='bold')
        ax1.set_xlabel('X (voxels)')
        ax1.set_ylabel('Y (voxels)')
        ax1.set_zlabel('Z (voxels)')
        ax1.view_init(elev=25, azim=45)

    # Second subplot: Original core structure
    ax2 = fig.add_subplot(122, projection='3d')

    # Pore part
    pore_mask = (cube == 1)
    # import pdb;pdb.set_trace()

    ax2.voxels(pore_mask, facecolors='blue', edgecolor='darkblue', alpha=0.8)

    # Matrix part
    matrix_mask = (cube == 2)
    ax2.voxels(matrix_mask, facecolors='red', edgecolor='darkred', alpha=0.3)

    ax2.set_xlabel('X (voxels)')
    ax2.set_ylabel('Y (voxels)')
    ax2.set_zlabel('Z (voxels)')
    ax2.set_title('Original 3D Core Structure', fontsize=12, fontweight='bold')
    ax2.view_init(elev=25, azim=45)

    plt.tight_layout()
    plt.savefig(f'{path}_fig1_simple_classification.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_comparison_with_size_legend(net, path, resolution_um=4.0):
    """
    绘制三类参数的4类饼状图、占比和分布曲线

    参数：
        net: 孔隙网络对象
        path: 保存路径前缀
        resolution_um: 分辨率（微米/体素）
    """
    # 获取所有分类数据
    connectivity_labels = classify_pores_by_connectivity_simple(net)
    size_labels, pore_diameters_um = classify_pores_by_size(net, resolution_um)
    throat_data = calculate_pore_throat_characteristics(net, resolution_um)
    throat_labels = throat_data['dominant_throat_class']

    # 获取网络基本信息
    coords = net.network['pore.coords']
    conns = net.network['throat.conns']
    n_pores = len(coords)

    print(f"Debug: n_pores={n_pores}, throat_labels_len={len(throat_labels)}")

    # 创建一个大图，包含3行2列的子图
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle('Pore Network Feature Analysis', fontsize=16, fontweight='bold')

    # ========== 第一行：连通性分析 ==========
    # 1.1 连通性饼状图
    ax1 = plt.subplot(3, 2, 1)
    conn_categories = ['isolated', 'dead_end', 'well_connected', 'highly_connected']
    conn_colors = ['#e74c3c', '#f39c12', '#3498db', '#9b59b6']

    # 计算连通性类别数量
    conn_counts = []
    conn_percentages = []
    for category in conn_categories:
        count = connectivity_labels.count(category)
        conn_counts.append(count)
        conn_percentages.append(count / len(connectivity_labels) * 100)

    # 检查是否有数据
    if sum(conn_counts) > 0:
        # 绘制饼状图（只显示有数据的类别）
        pie_data = []
        pie_labels = []
        pie_colors = []
        for i, (count, category, color) in enumerate(zip(conn_counts, conn_categories, conn_colors)):
            if count > 0:
                pie_data.append(count)
                pie_labels.append(category)
                pie_colors.append(color)

        if pie_data:
            wedges, texts, autotexts = ax1.pie(
                pie_data,
                labels=pie_labels,
                colors=pie_colors,
                autopct='%1.1f%%',
                startangle=90,
                wedgeprops=dict(edgecolor='white', linewidth=1)
            )

            # 美化百分比文本
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        else:
            ax1.text(0.5, 0.5, 'No connectivity data',
                     ha='center', va='center', fontsize=12)
    else:
        ax1.text(0.5, 0.5, 'No pores found',
                 ha='center', va='center', fontsize=12)

    ax1.set_title('Connectivity Classification', fontsize=14, fontweight='bold')
    ax1.axis('equal')

    # 1.2 连通性分布曲线
    ax2 = plt.subplot(3, 2, 2)

    # 计算配位数
    coordination = np.zeros(n_pores)
    for conn in conns:
        coordination[conn[0]] += 1
        coordination[conn[1]] += 1

    # 绘制直方图
    if len(coordination) > 0 and np.any(coordination > 0):
        max_coord = int(coordination.max())
        bins = np.arange(-0.5, max_coord + 1.5, 1)

        # 分组绘制
        for i, category in enumerate(conn_categories):
            mask = np.array([cl == category for cl in connectivity_labels])
            if mask.any() and np.any(coordination[mask] >= 0):
                ax2.hist(coordination[mask], bins=bins, alpha=0.7,
                         color=conn_colors[i], label=category,
                         edgecolor='black', linewidth=0.5)

    ax2.set_xlabel('Coordination Number')
    ax2.set_ylabel('Number of Pores')
    ax2.set_title('Coordination Number Distribution by Connectivity Type', fontsize=12)
    if len(ax2.get_legend_handles_labels()[0]) > 0:
        ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ========== 第二行：孔隙尺寸分析 ==========
    # 2.1 孔隙尺寸饼状图
    ax3 = plt.subplot(3, 2, 3)
    size_categories = ['nanopore', 'micropore', 'mesopore', 'macropore']
    size_colors = ['#1abc9c', '#2ecc71', '#3498db', '#9b59b6']

    # 计算孔隙尺寸类别数量
    size_counts = []
    for category in size_categories:
        count = size_labels.count(category)
        size_counts.append(count)

    # 检查是否有数据
    if sum(size_counts) > 0:
        # 绘制饼状图（只显示有数据的类别）
        pie_data = []
        pie_labels = []
        pie_colors = []
        for i, (count, category, color) in enumerate(zip(size_counts, size_categories, size_colors)):
            if count > 0:
                pie_data.append(count)
                pie_labels.append(category)
                pie_colors.append(color)

        if pie_data:
            wedges, texts, autotexts = ax3.pie(
                pie_data,
                labels=pie_labels,
                colors=pie_colors,
                autopct='%1.1f%%',
                startangle=90,
                wedgeprops=dict(edgecolor='white', linewidth=1)
            )

            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        else:
            ax3.text(0.5, 0.5, 'No size data',
                     ha='center', va='center', fontsize=12)
    else:
        ax3.text(0.5, 0.5, 'No size data',
                 ha='center', va='center', fontsize=12)

    ax3.set_title('Pore Size Classification', fontsize=14, fontweight='bold')
    ax3.axis('equal')

    # 2.2 孔隙尺寸分布曲线
    ax4 = plt.subplot(3, 2, 4)

    if len(pore_diameters_um) > 0 and np.any(pore_diameters_um > 0):
        # 创建合适的bins
        non_zero_diams = pore_diameters_um[pore_diameters_um > 0]
        if len(non_zero_diams) > 0:
            min_diam = non_zero_diams.min()
            max_diam = non_zero_diams.max()

            # 使用对数bins
            if max_diam > min_diam * 10 and min_diam > 0:
                bins = np.logspace(np.log10(min_diam), np.log10(max_diam), 20)
                ax4.set_xscale('log')
            else:
                bins = np.linspace(min_diam, max_diam, 20)

            # 分组绘制直方图
            for i, category in enumerate(size_categories):
                mask = np.array([sl == category for sl in size_labels])
                if mask.any() and np.any(pore_diameters_um[mask] > 0):
                    ax4.hist(pore_diameters_um[mask], bins=bins, alpha=0.7,
                             color=size_colors[i], label=category,
                             edgecolor='black', linewidth=0.5)

    ax4.set_xlabel('Pore Diameter (µm)')
    ax4.set_ylabel('Number of Pores')
    ax4.set_title('Pore Size Distribution by Size Type', fontsize=12)
    if len(ax4.get_legend_handles_labels()[0]) > 0:
        ax4.legend()
    ax4.grid(True, alpha=0.3)

    # ========== 第三行：喉道类型分析 ==========
    # 3.1 喉道类型饼状图
    ax5 = plt.subplot(3, 2, 5)
    throat_categories = ['nano_throat_dominant', 'micro_throat_dominant',
                         'meso_throat_dominant', 'macro_throat_dominant']
    throat_colors = ['#e67e22', '#f1c40f', '#1abc9c', '#3498db']

    # 计算喉道类型数量 - 需要处理长度不匹配的问题
    throat_counts = []
    for category in throat_categories:
        # 安全计数
        count = 0
        for label in throat_labels:
            if label == category:
                count += 1
        throat_counts.append(count)

    # 检查是否有数据
    if sum(throat_counts) > 0:
        # 绘制饼状图（只显示有数据的类别）
        pie_data = []
        pie_labels = []
        pie_colors = []
        for i, (count, category, color) in enumerate(zip(throat_counts, throat_categories, throat_colors)):
            if count > 0:
                pie_data.append(count)
                pie_labels.append(category.replace('_', '\n'))
                pie_colors.append(color)

        if pie_data:
            wedges, texts, autotexts = ax5.pie(
                pie_data,
                labels=pie_labels,
                colors=pie_colors,
                autopct='%1.1f%%',
                startangle=90,
                wedgeprops=dict(edgecolor='white', linewidth=1)
            )

            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        else:
            ax5.text(0.5, 0.5, 'No throat data',
                     ha='center', va='center', fontsize=12)
    else:
        ax5.text(0.5, 0.5, 'No throat data',
                 ha='center', va='center', fontsize=12)

    ax5.set_title('Throat Type Classification', fontsize=14, fontweight='bold')
    ax5.axis('equal')

    # 3.2 喉道尺寸分布曲线
    ax6 = plt.subplot(3, 2, 6)

    if 'max_throat_size_um' in throat_data and len(throat_data['max_throat_size_um']) > 0:
        throat_sizes = throat_data['max_throat_size_um']

        # 确保throat_labels和throat_sizes长度匹配
        if len(throat_labels) == len(throat_sizes):
            # 只考虑有喉道的孔隙
            valid_indices = [i for i, size in enumerate(throat_sizes) if size > 0]
            if valid_indices:
                throat_sizes_valid = throat_sizes[valid_indices]
                valid_labels = [throat_labels[i] for i in valid_indices]

                # 创建合适的bins
                min_size = max(0.001, throat_sizes_valid.min())
                max_size = throat_sizes_valid.max()

                # 使用对数bins（喉道尺寸范围可能很大）
                if max_size > min_size * 100 and min_size > 0:
                    bins = np.logspace(np.log10(min_size), np.log10(max_size), 20)
                    ax6.set_xscale('log')
                else:
                    bins = np.linspace(min_size, max_size, 20)

                # 分组绘制直方图
                for i, category in enumerate(throat_categories):
                    mask = np.array([tl == category for tl in valid_labels])
                    if mask.any() and np.any(throat_sizes_valid[mask] > 0):
                        ax6.hist(throat_sizes_valid[mask], bins=bins, alpha=0.7,
                                 color=throat_colors[i], label=category.replace('_', '\n'),
                                 edgecolor='black', linewidth=0.5)

    ax6.set_xlabel('Maximum Throat Size (µm)')
    ax6.set_ylabel('Number of Pores')
    ax6.set_title('Throat Size Distribution by Type', fontsize=12)
    if len(ax6.get_legend_handles_labels()[0]) > 0:
        ax6.legend()
    ax6.grid(True, alpha=0.3)

    # 添加汇总信息文本框
    # 安全计算平均值
    avg_coordination = np.mean(coordination) if len(coordination) > 0 else 0
    avg_pore_diameter = np.mean(pore_diameters_um[pore_diameters_um > 0]) if len(pore_diameters_um) > 0 else 0

    # 计算喉道尺寸平均值
    avg_throat_size = 0
    if 'max_throat_size_um' in throat_data and len(throat_data['max_throat_size_um']) > 0:
        throat_sizes_nonzero = throat_data['max_throat_size_um'][throat_data['max_throat_size_um'] > 0]
        if len(throat_sizes_nonzero) > 0:
            avg_throat_size = np.mean(throat_sizes_nonzero)

    summary_text = f"""
    Summary Statistics:
    Total Pores: {n_pores}
    Total Throats: {len(conns)}
    Avg Coordination: {avg_coordination:.2f}
    Avg Pore Diameter: {avg_pore_diameter:.2f} µm
    Avg Throat Size: {avg_throat_size:.4f} µm
    """

    fig.text(0.02, 0.98, summary_text, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 为标题留出空间
    plt.savefig(f'{path}_fig2_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 另外创建一个简化的汇总图（可选）
    # 注意：需要确保所有数组长度匹配
    try:
        create_summary_figure(net, path, resolution_um,
                              connectivity_labels, size_labels, throat_labels,
                              pore_diameters_um, throat_data)
    except Exception as e:
        print(f"Warning: Could not create summary figure: {e}")

# 同时修复calculate_pore_throat_characteristics函数，确保返回所有孔隙的标签
def calculate_pore_throat_characteristics(net, resolution_um=4.0):
    """
    Calculate throat-related characteristics for each pore.
    修复版：确保为所有孔隙返回标签，即使没有喉道
    """
    throat_diameters_pixels = net.network['throat.equivalent_diameter']
    throat_conns = net.network['throat.conns']
    throat_diameters_um = throat_diameters_pixels * resolution_um

    n_pores = len(net.network['pore.coords'])

    # Initialize arrays to store throat characteristics for each pore
    max_throat_size = np.zeros(n_pores)
    mean_throat_size = np.zeros(n_pores)
    throat_size_labels = ['no_throat'] * n_pores  # 为所有孔隙初始化标签

    # Temporary lists to store all throat sizes for each pore
    pore_throat_sizes = [[] for _ in range(n_pores)]

    # Collect throat sizes for each pore
    for i, conn in enumerate(throat_conns):
        pore1, pore2 = conn[0], conn[1]
        throat_size = throat_diameters_um[i]
        pore_throat_sizes[pore1].append(throat_size)
        pore_throat_sizes[pore2].append(throat_size)

    # Calculate statistics for each pore
    for i in range(n_pores):
        if pore_throat_sizes[i]:  # If pore has connected throats
            max_throat_size[i] = max(pore_throat_sizes[i])
            mean_throat_size[i] = np.mean(pore_throat_sizes[i])

            # Determine dominant throat size class based on maximum throat
            if max_throat_size[i] < 0.063:
                throat_size_labels[i] = 'nano_throat_dominant'
            elif 0.063 <= max_throat_size[i] < 0.62:
                throat_size_labels[i] = 'micro_throat_dominant'
            elif 0.62 <= max_throat_size[i] <= 7.5:
                throat_size_labels[i] = 'meso_throat_dominant'
            else:
                throat_size_labels[i] = 'macro_throat_dominant'

    return {
        'max_throat_size_um': max_throat_size,
        'mean_throat_size_um': mean_throat_size,
        'dominant_throat_class': throat_size_labels
    }

def create_summary_figure(net, path, resolution_um,
                          connectivity_labels, size_labels, throat_labels,
                          pore_diameters_um, throat_data):
    """
    创建一个简化的汇总图，显示三类特征的对比
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 准备数据
    conn_categories = ['isolated', 'dead_end', 'well_connected', 'highly_connected']
    conn_colors = ['#e74c3c', '#f39c12', '#3498db', '#9b59b6']

    size_categories = ['nanopore', 'micropore', 'mesopore', 'macropore']
    size_colors = ['#1abc9c', '#2ecc71', '#3498db', '#9b59b6']

    throat_categories = ['nano_throat_dominant', 'micro_throat_dominant',
                         'meso_throat_dominant', 'macro_throat_dominant']
    throat_colors = ['#e67e22', '#f1c40f', '#1abc9c', '#3498db']

    # 计算比例
    conn_percentages = [connectivity_labels.count(cat) / len(connectivity_labels) * 100
                        for cat in conn_categories]
    size_percentages = [size_labels.count(cat) / len(size_labels) * 100
                        for cat in size_categories]
    throat_percentages = [throat_labels.count(cat) / len(throat_labels) * 100
                          for cat in throat_categories]

    # 1. 连通性条形图
    bars1 = axes[0].bar(conn_categories, conn_percentages, color=conn_colors, alpha=0.8)
    axes[0].set_title('Connectivity Distribution', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Percentage (%)')
    axes[0].set_ylim(0, max(conn_percentages) * 1.2)
    axes[0].tick_params(axis='x', rotation=45)

    # 添加数值标签
    for bar, percentage in zip(bars1, conn_percentages):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2., height + 1,
                     f'{percentage:.1f}%', ha='center', va='bottom', fontsize=10)

    # 2. 孔隙尺寸条形图
    bars2 = axes[1].bar(size_categories, size_percentages, color=size_colors, alpha=0.8)
    axes[1].set_title('Pore Size Distribution', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Percentage (%)')
    axes[1].set_ylim(0, max(size_percentages) * 1.2)
    axes[1].tick_params(axis='x', rotation=45)

    for bar, percentage in zip(bars2, size_percentages):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2., height + 1,
                     f'{percentage:.1f}%', ha='center', va='bottom', fontsize=10)

    # 3. 喉道类型条形图
    bars3 = axes[2].bar(throat_categories, throat_percentages, color=throat_colors, alpha=0.8)
    axes[2].set_title('Throat Type Distribution', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Percentage (%)')
    axes[2].set_ylim(0, max(throat_percentages) * 1.2)
    axes[2].tick_params(axis='x', rotation=45)

    for bar, percentage in zip(bars3, throat_percentages):
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width() / 2., height + 1,
                     f'{percentage:.1f}%', ha='center', va='bottom', fontsize=10)

    # 添加总标题
    coords = net.network['pore.coords']
    conns = net.network['throat.conns']

    fig.suptitle(f'Pore Network Feature Summary (Total: {len(coords)} pores, {len(conns)} throats)',
                 fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{path}_fig3_summary_bars.png', dpi=300, bbox_inches='tight')
    plt.close()


# 在主函数中调用
def plot_network_statistics(net, cube, filepath_fig,resolution_um):
    """
    显示统计信息
    """
    # 调用第一个图（保持原样）
    plot_comparison(cube, net, filepath_fig)

    # 调用第二个图（新增的综合分析图）
    # 需要确定分辨率，这里假设为4.0，你可以根据实际情况调整
    # resolution_um = 4.0  # 或者从其他地方获取
    plot_comparison_with_size_legend(net, filepath_fig, resolution_um)