# Reference: The code has been modified from the LDM repo: https://https://github.com/yccyenchicheng/SDFusion
# Code for the training of various models (e.g., VQ-VAE, MOF-Constructor, Unconditional & Conditional MOFFUSION)
import os
import numpy as np
import pandas as pd
import porespy as ps
import openpnm as op
from descriptors_pore_extract import *
from network_statistics import *
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re
import warnings

warnings.filterwarnings("ignore")

# 设置中文字体为微软雅黑
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

from skimage.util import view_as_windows
def sliding_window_subcubes(voxel, subcube_size=32, stride=32):
    """滑动窗口生成子立方体"""
    return view_as_windows(voxel, (subcube_size, subcube_size, subcube_size), stride)

def calculate_pore_throat_characteristics(net, resolution_um=4.0):
    """
    Calculate throat-related characteristics for each pore.

    Returns:
    --------
    pore_throat_data : dict
        Dictionary containing various throat characteristics for each pore.
    """

    throat_diameters_pixels = net.network['throat.equivalent_diameter']
    throat_conns = net.network['throat.conns']
    throat_diameters_um = throat_diameters_pixels * resolution_um

    n_pores = len(net.network['pore.coords'])

    # Initialize arrays to store throat characteristics for each pore
    max_throat_size = np.zeros(n_pores)  # Maximum throat size connected to the pore
    mean_throat_size = np.zeros(n_pores)  # Mean throat size connected to the pore
    throat_size_labels = []  # Dominant throat size class for the pore

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
                throat_size_labels.append('nano_throat_dominant')
            elif 0.063 <= max_throat_size[i] < 0.62:
                throat_size_labels.append('micro_throat_dominant')
            elif 0.62 <= max_throat_size[i] <= 7.5:
                throat_size_labels.append('meso_throat_dominant')
            else:
                throat_size_labels.append('macro_throat_dominant')
        # else:  # Isolated pore with no throats
        #     max_throat_size[i] = 0
        #     mean_throat_size[i] = 0
        #     throat_size_labels.append('no_throat')

    return {
        'max_throat_size_um': max_throat_size,
        'mean_throat_size_um': mean_throat_size,
        'dominant_throat_class': throat_size_labels
    }


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

    return topology_types, coordination


def extract_complete_pore_features(net, resolution_um=4.0):
    """
    Extract complete pore features including size, connectivity, and throat characteristics.

    Returns:
    --------
    feature_proportions : dict
        Dictionary containing proportions of different feature types
    """
    # 1. 孔隙连通性分类
    connectivity_labels, coordination = classify_pores_by_connectivity_simple(net)

    # 2. 孔隙尺寸分类
    size_labels, pore_diameters_um = classify_pores_by_size(net, resolution_um)

    # 3. 喉道特征计算
    throat_data = calculate_pore_throat_characteristics(net, resolution_um)

    print(f"Processing pore features:")
    print(f"  Connectivity labels: {len(connectivity_labels)} pores")
    print(f"  Size labels: {len(size_labels)} pores")
    print(f"  Throat labels: {len(throat_data['dominant_throat_class'])} pores")

    # 检查是否存在no_throat类别
    throat_labels = throat_data['dominant_throat_class']

    # 计算各类别的比例
    feature_proportions = {}

    # 1. 连通性类型比例（4类）
    connectivity_labels_series = pd.Series(connectivity_labels)
    connectivity_counts = connectivity_labels_series.value_counts(normalize=True) * 100

    print(f"\n[连通性分类] - 4类:")
    conn_categories = ['isolated', 'dead_end', 'well_connected', 'highly_connected']
    for conn_type in conn_categories:
        proportion = connectivity_counts.get(conn_type, 0)
        feature_proportions[f'prop_{conn_type}'] = proportion
        count = (connectivity_labels_series == conn_type).sum()
        print(f"  {conn_type:15s}: {count:3d} pores ({proportion:5.1f}%)")

    # 2. 孔隙尺寸类型比例（4类）
    size_labels_series = pd.Series(size_labels)
    size_counts = size_labels_series.value_counts(normalize=True) * 100

    print(f"\n[孔隙尺寸分类] - 4类:")
    size_categories = ['nanopore', 'micropore', 'mesopore', 'macropore']
    for size_type in size_categories:
        proportion = size_counts.get(size_type, 0)
        feature_proportions[f'prop_{size_type}'] = proportion
        count = (size_labels_series == size_type).sum()
        print(f"  {size_type:10s}: {count:3d} pores ({proportion:5.1f}%)")

    # 3. 喉道类型比例（4类）
    throat_labels_series = pd.Series(throat_labels)
    throat_counts = throat_labels_series.value_counts(normalize=True) * 100

    print(f"\n[喉道类型分类] - 4类:")
    throat_categories = ['nano_throat_dominant', 'micro_throat_dominant',
                         'meso_throat_dominant', 'macro_throat_dominant']
    for throat_type in throat_categories:
        proportion = throat_counts.get(throat_type, 0)
        feature_proportions[f'prop_{throat_type}'] = proportion
        count = (throat_labels_series == throat_type).sum()
        print(f"  {throat_type:20s}: {count:3d} pores ({proportion:5.1f}%)")

    # 检查是否有其他喉道类型
    other_throat_types = set(throat_labels) - set(throat_categories)
    if other_throat_types:
        print(f"\nWarning: Found unexpected throat types: {other_throat_types}")
        for throat_type in other_throat_types:
            count = (throat_labels_series == throat_type).sum()
            proportion = throat_counts.get(throat_type, 0)
            print(f"  {throat_type:20s}: {count:3d} pores ({proportion:5.1f}%)")

    # 4. 计算一些汇总统计量
    print(f"\n[汇总统计]:")

    # 平均配位数
    avg_coordination = np.mean(coordination) if len(coordination) > 0 else 0
    feature_proportions['avg_coordination'] = avg_coordination
    print(f"  平均配位数: {avg_coordination:.2f}")

    # 平均孔隙直径
    avg_pore_diameter = np.mean(pore_diameters_um) if len(pore_diameters_um) > 0 else 0
    feature_proportions['avg_pore_diameter_um'] = avg_pore_diameter
    print(f"  平均孔隙直径: {avg_pore_diameter:.2f} µm")

    # 平均喉道尺寸
    if len(throat_data['mean_throat_size_um']) > 0:
        avg_throat_size = np.mean(throat_data['mean_throat_size_um'])
        feature_proportions['avg_throat_size_um'] = avg_throat_size
        print(f"  平均喉道尺寸: {avg_throat_size:.4f} µm")

    # 最大喉道尺寸
    if len(throat_data['max_throat_size_um']) > 0:
        max_throat_size = np.max(throat_data['max_throat_size_um'])
        feature_proportions['max_throat_size_um'] = max_throat_size
        print(f"  最大喉道尺寸: {max_throat_size:.4f} µm")

    return feature_proportions


def analyze_simple_connectivity(net, resolution_um=4.0):
    """
    Simplified connectivity analysis with complete feature extraction
    """
    # 使用完整的特征提取函数
    feature_proportions = extract_complete_pore_features(net, resolution_um)

    coords = net.network['pore.coords']
    conns = net.network['throat.conns']

    print("=== Simplified Connectivity Analysis ===")
    print(f"Total pores: {len(coords)}")
    print(f"Total throats: {len(conns)}")
    print(f"Average coordination number: {feature_proportions['avg_coordination']:.2f}")

    # 返回feature_proportions，其他两个值设为None
    return None, None, feature_proportions

def extract_pore_network(cube, voxel_size=1.0):
    """
    Extract pore network (ensure extraction for pore phase)
    """
    # Ensure correct phase orientation
    cube_corrected = cube + 1
    print(f"Unique values in cube: {np.unique(cube_corrected)}")

    # Verify porosity
    porosity = ps.metrics.porosity(cube_corrected - 1)
    print(f"Porosity for network extraction: {porosity:.3f}")

    try:
        # Use SNOW2 algorithm to extract pore network
        phases = cube_corrected  # Now: pore=2, matrix=1

        net = ps.networks.snow2(
            phases=phases,
            phase_alias={2: 'pore'},  # Specify label 2 represents pore phase
            voxel_size=voxel_size,
        )

        return net, cube_corrected

    except Exception as e:
        print(f"Network extraction failed: {e}")
        return None, cube_corrected


if __name__ == "__main__":
    # Parameter settings
    res = 256
    data_paths = './data/'
    output = f'./sub_data/resolution_{res}'
    output1 = f'./sub_txt/resolution_{res}'
    output2 = f'./visual_fig/resolution_{res}'
    os.makedirs(output, exist_ok=True)
    os.makedirs(output1, exist_ok=True)
    os.makedirs(output2, exist_ok=True)

    # 这个是储存，由体计算出来的：
    # Create log file
    #保存2个txt；

    # 创建两个日志文件
    summary_file = os.path.join(output1, 'summary.txt')
    topology_file = os.path.join(output1, 'topology.txt')
    # 初始化summary文件（完整结构参数）
    with open(summary_file, 'w') as f:
        header = "name\t"
        # 从extract_structure_3d返回的keys中选择要保存的
        header += "Porosity\tMean_Pore_Radius\tFractal_Dimension\tMean_Throat_Radius\t"
        header += "Mean_Coordination\tSpecific_Surface_Pores\t"
        header += "Sphericity\tAspect_Ratio\t"
        header += "Tortuosity\n"
        f.write(header)


    # 初始化topology文件（连通性等比例参数）
    with open(topology_file, 'w') as f:
        header = "name\t"
        # 连通性比例
        header += "prop_isolated\tprop_dead_end\tprop_well_connected\tprop_highly_connected\t"
        # 孔隙尺寸比例
        header += "prop_nanopore\tprop_micropore\tprop_mesopore\tprop_macropore\t"
        # 喉道主导类型比例
        header += "prop_nano_throat_dominant\tprop_micro_throat_dominant\t"
        header += "prop_meso_throat_dominant\tprop_macro_throat_dominant\t"
        # 汇总统计信息
        header += "avg_coordination\tavg_pore_diameter_um\tavg_throat_size_um\tmax_throat_size_um\n"

        f.write(header)

    # 处理每个数据文件
    data_files = os.listdir(data_paths)

    for filename in data_files:
        print(f"\n{'=' * 50}")
        print(f"Processing file: {filename}")
        print(f"{'=' * 50}")

        # 提取样品名称
        match = re.match(r'([cCsS]\d+)', filename)
        if match:
            name = match.group(1).lower()
        else:
            print(f"Cannot match filename: {filename}")
            continue

        # 设置分辨率（根据样品类型）
        if name.startswith('c'):  # 碳酸盐岩
            resolution_um = 4.0
            dim_size = 1001  # c1.raw 和 c2.raw 是 1001x1001x1001
        elif name.startswith('s'):  # 砂岩
            resolution_um = 2.5
            dim_size = 1000  # s1.raw 和 s2.raw 是 1000x1000x1000
        else:
            print(f"Unknown sample type: {name}")
            continue

        print(f"Sample: {name}, Resolution: {resolution_um} um/voxel, Size: {dim_size}x{dim_size}x{dim_size}")

        # 读取数据
        data_path = os.path.join(data_paths, filename)
        data = np.fromfile(data_path, dtype=np.uint8).reshape((dim_size,dim_size,dim_size))
        print(data.shape)

        # Generate subcubes
        subcubes = sliding_window_subcubes(data, res, 100)
        subcubes = subcubes.reshape(-1, res, res, res)
        print(f"Generated {len(subcubes)} subcubes")

        for i, cube in enumerate(subcubes):
            print(f"\nProcessing subcube {i}:")

            cube = 1 - cube  # Read in, 1 is pore; 0 is matrix

            original_porosity = ps.metrics.porosity(cube)
            print(f"Original porosity: {original_porosity:.3f}")

            # Porosity filtering
            if 0.05 <= original_porosity <= 0.5:
                # Extract pore network
                net, corrected_cube = extract_pore_network(cube)
                pore_structure_result = extract_structure_3d(cube,resolution_um)

                if net is not None:
                    # Get network parameters
                    node_count = net.network['pore.coords'].shape[0]
                    edge_count = net.network['throat.conns'].shape[0]

                    filename = f"{name}_{i}"
                    # Calculate pore proportions using the new function
                    _,_,feature_proportions = analyze_simple_connectivity(net,resolution_um=resolution_um)
                    # 保存结构参数到summary文件
                    with open(summary_file, 'a') as f:
                        f.write(f"{filename}\t")
                        # 写入结构参数
                        keys_to_save = [
                            'Porosity', 'Mean_Pore_Radius', 'Fractal_Dimension',
                            'Mean_Throat_Radius', 'Mean_Coordination',
                            'Specific_Surface_Pores',
                            'Sphericity', 'Aspect_Ratio', 'Tortuosity'
                        ]
                        for key in keys_to_save:
                            if key in pore_structure_result:
                                val = pore_structure_result[key]
                                if isinstance(val, (list, tuple)):
                                    f.write(f"{len(val)}\t")
                                else:
                                    f.write(f"{val:.4f}\t")

                        f.write("\n")

                                # 保存拓扑参数到topology文件
                    with open(topology_file, 'a') as f:
                        f.write(f"{filename}\t")
                        # 写入连通性比例
                        for prop_key in ['prop_isolated', 'prop_dead_end',
                                         'prop_well_connected','prop_highly_connected']:
                            f.write(f"{feature_proportions.get(prop_key, 0):.3f}\t")

                        # 写入孔隙尺寸比例
                        for prop_key in ['prop_nanopore', 'prop_micropore',
                                         'prop_mesopore','prop_macropore']:
                            f.write(f"{feature_proportions.get(prop_key, 0):.3f}\t")

                        # 写入喉道类型比例
                        throat_keys = ['prop_nano_throat_dominant', 'prop_micro_throat_dominant',
                                       'prop_meso_throat_dominant', 'prop_macro_throat_dominant']
                        for prop_key in throat_keys:
                            f.write(f"{feature_proportions.get(prop_key, 0):.3f}\t")

                        # 写入汇总统计信息
                        stat_keys = ['avg_coordination', 'avg_pore_diameter_um',
                                     'avg_throat_size_um', 'max_throat_size_um']
                        for key in stat_keys:
                            val = feature_proportions.get(key, 0)
                            if key.endswith('_um'):
                                f.write(f"{val:.4f}\t")  # 尺寸信息保留4位小数
                            else:
                                f.write(f"{val:.3f}\t")  # 其他信息保留3位小数

                        f.write("\n")

                    # Save results
                    filename_npy = f'{name}_{i}+{node_count}+{edge_count}.npy'

                    filepath = os.path.join(output, filename_npy)
                    # import pdb;pdb.set_trace()
                    np.save(filepath, (corrected_cube-1).astype(np.uint8))
                    #1 p; 0 m

                    file_fig = f'{name}_{i}_por{original_porosity:.4f}_node{node_count}_edge{edge_count}_fig'
                    # 1 is matrix , 2 is pore
                    # 显示对比图
                    filepath_fig = os.path.join(output2, file_fig)

                    plot_network_statistics(net, corrected_cube,filepath_fig, resolution_um)

                    print(f"Successfully processed subcube {i}, features saved.")

                else:
                    print("Network extraction failed, skipping subcube")
            else:
                print(f"孔隙度 {original_porosity:.3f} 不在目标范围内(0.05-0.5)，跳过")

    print(f"\n处理完成！结果保存在: {output}")

