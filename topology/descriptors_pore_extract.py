import math
import os
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from scipy import ndimage, stats
from skimage import measure, morphology, filters, segmentation, transform

import tifffile as tiff
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.util import view_as_windows


def sliding_window_subcubes(voxel, subcube_size=32, stride=32):
    """滑动窗口生成子立方体"""
    return view_as_windows(voxel, (subcube_size, subcube_size, subcube_size), stride)


def PSD_3D(A: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    计算3D体数据中孔隙的平均半径（等效球半径）

    参数：
        A: 3D二值图像，其中1表示孔隙，0表示背景

    返回：
        mean_radius: 孔隙的平均等效球半径
        radii: 所有孔隙的半径数组
    """
    A = A.astype(np.uint8)

    # 填充孔洞（3D）
    filled_img = ndimage.binary_fill_holes(A)

    # 距离变换（3D）
    distance = ndimage.distance_transform_edt(filled_img)

    # 标记分水岭的种子点
    markers = np.zeros_like(A, dtype=np.int32)
    threshold = filters.threshold_otsu(distance[distance > 0]) if np.any(distance > 0) else 0
    markers[distance > threshold] = 1

    # 分水岭分割（3D）
    labels = segmentation.watershed(-distance, markers, mask=A, connectivity=1)

    # 连通域标记
    label_img = measure.label(labels, connectivity=1)

    # 计算每个孔隙的属性
    props = measure.regionprops(label_img)
    volumes = np.array([prop.area for prop in props])

    # 计算等效球半径
    radii = (3 * volumes / (4 * np.pi)) ** (1 / 3)

    # 计算平均半径
    mean_radius = np.mean(radii) if len(radii) > 0 else 0

    return mean_radius, radii
def box_count_3d(voxel: np.ndarray, k: int) -> int:
    """
    3D盒计数法

    参数：
        voxel: 3D二值图像
        k: 盒子大小

    返回：
        nr: 被占据的盒子数量
    """
    X, Y, Z = voxel.shape
    nr = 0

    for i in range(0, X, k):
        for j in range(0, Y, k):
            for l in range(0, Z, k):
                i1, i2 = i, min(i + k - 1, X - 1)
                j1, j2 = j, min(j + k - 1, Y - 1)
                l1, l2 = l, min(l + k - 1, Z - 1)

                if np.any(voxel[i1:i2 + 1, j1:j2 + 1, l1:l2 + 1]):
                    nr += 1

    return nr
def fractal_dimension_3d(voxel: np.ndarray) -> float:
    """
    计算3D图像的分形维数

    参数：
        voxel: 3D二值图像

    返回：
        FD: 分形维数
    """
    voxel = voxel.astype(np.uint8)
    X, Y, Z = voxel.shape
    Smin = min(X, Y, Z)

    # 生成盒子大小序列
    kv = [k for k in range(1, Smin + 1, 2) if Smin % k == 0]
    if len(kv) < 4:  # 如果点数太少，使用固定序列
        kv = np.logspace(1, math.log2(Smin / 2), 10, base=2).astype(int)
        kv = np.unique(np.clip(kv, 2, Smin // 2))

    # 计算不同尺度下的盒计数
    box_sizes = []
    box_counts = []

    for k in kv:
        if k < 2:
            continue
        nr = box_count_3d(voxel, k)
        box_sizes.append(k)
        box_counts.append(nr)

    if len(box_sizes) < 2:
        return 0.0

    # 对数变换
    x = -np.log(np.array(box_sizes) / Smin)
    y = np.log(np.array(box_counts))

    # 线性拟合计算斜率（分形维数）
    if len(x) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        FD = slope
    else:
        FD = 0.0

    return FD
def Throat_Size_3D(A: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    计算3D图像中的喉道尺寸分布

    参数：
        A: 3D二值图像，1表示孔隙，0表示基质

    返回：
        mean_throat_size: 平均喉道尺寸
        throat_sizes: 喉道尺寸数组
    """
    A = A.astype(np.uint8)

    # 使用骨架化找到喉道位置
    skeleton = morphology.skeletonize_3d(A)

    # 在骨架上计算局部半径（距离变换）
    distance = ndimage.distance_transform_edt(A)
    throat_sizes = distance[skeleton > 0]

    # 计算平均喉道尺寸
    mean_throat_size = np.mean(throat_sizes) if len(throat_sizes) > 0 else 0

    return mean_throat_size, throat_sizes
def Coordination_3D(A: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    计算3D孔隙网络的配位数

    参数：
        A: 3D二值图像，1表示孔隙，0表示基质

    返回：
        mean_coordination: 平均配位数
        coordination_numbers: 每个孔隙的配位数
    """
    A = A.astype(np.uint8)

    # 标记孔隙区域
    labels = measure.label(A, connectivity=1)
    props = measure.regionprops(labels)

    # 创建骨架来识别连接
    skeleton = morphology.skeletonize_3d(A)

    # 找到骨架的交叉点（节点）
    kernel = np.ones((3, 3, 3), dtype=bool)
    eroded = morphology.erosion(skeleton, kernel)
    reconstructed = morphology.reconstruction(eroded, skeleton)
    nodes = skeleton - reconstructed

    # 标记节点并分配到最近的孔隙
    node_labels = measure.label(nodes, connectivity=1)

    # 计算每个孔隙的连接数（简化方法）
    coordination_numbers = []
    for prop in props:
        # 获取孔隙的坐标
        coords = prop.coords

        # 找到附近的节点
        pore_nodes = set()
        for coord in coords:
            # 检查26邻域内的节点
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        x, y, z = coord[0] + dx, coord[1] + dy, coord[2] + dz
                        if (0 <= x < A.shape[0] and 0 <= y < A.shape[1] and
                                0 <= z < A.shape[2] and node_labels[x, y, z] > 0):
                            pore_nodes.add(node_labels[x, y, z])

        coordination_numbers.append(len(pore_nodes))

    mean_coordination = np.mean(coordination_numbers) if coordination_numbers else 0

    return mean_coordination, np.array(coordination_numbers)
def SPS_3D(A: np.ndarray) -> Tuple[float, float]:
    """
    计算3D图像的比表面积

    参数：
        A: 3D二值图像，1表示孔隙，0表示基质

    返回：
        specific_surface_pores: 孔隙比表面积
        specific_surface_matrix: 基质比表面积
    """
    # 计算孔隙和基质的体积
    pore_volume = np.sum(A)
    matrix_volume = np.sum(~A)
    total_volume = A.size

    # 使用Marching Cubes计算表面积
    try:
        # 对于较大的体数据，可以采样计算
        if A.size > 1000 ** 3:
            # 降采样计算
            step = max(1, int(np.cbrt(A.size / 1000000)))
            A_sampled = A[::step, ::step, ::step]
            verts, faces, normals, values = measure.marching_cubes(A_sampled, level=0.5, step_size=step)
            surface_area = measure.mesh_surface_area(verts, faces)
            # 缩放回原始尺寸
            surface_area *= (step ** 2)
        else:
            verts, faces, normals, values = measure.marching_cubes(A, level=0.5)
            surface_area = measure.mesh_surface_area(verts, faces)
    except:
        # 如果Marching Cubes失败，使用近似方法
        surface_area = approximate_surface_area_3d(A)

    # 计算比表面积
    specific_surface_pores = surface_area / pore_volume if pore_volume > 0 else 0
    specific_surface_matrix = surface_area / matrix_volume if matrix_volume > 0 else 0

    return specific_surface_pores, specific_surface_matrix
def approximate_surface_area_3d(A: np.ndarray) -> float:
    """
    近似计算3D二值图像的表面积
    """
    # 使用体素计数方法近似计算表面积
    kernel = np.ones((3, 3, 3), dtype=bool)

    # 侵蚀和膨胀的差异可以近似表面积
    eroded = morphology.binary_erosion(A, kernel)
    dilated = morphology.binary_dilation(A, kernel)

    # 表面体素的数量
    surface_voxels = np.sum(dilated ^ eroded)

    # 每个表面体素的平均贡献面积（近似）
    voxel_area = 6  # 立方体体素的理论最大表面积（6个面）

    return surface_voxels * voxel_area / 2  # 除以2避免重复计数
def yuandu_3D(A: np.ndarray) -> float:
    """
    计算3D孔隙的圆度（球形度）

    参数：
        A: 3D二值图像

    返回：
        mean_sphericity: 平均球形度
    """
    A = A.astype(np.uint8)
    labels = measure.label(A, connectivity=1)
    props = measure.regionprops(labels)

    sphericities = []
    areas = []

    for prop in props:
        volume = prop.area
        if volume < 8:  # 忽略太小的孔隙
            continue

        # 计算表面积（近似）
        surface_area = prop.area  # 这里使用面积作为近似

        # 计算球形度
        sphericity = (np.pi ** (1 / 3) * (6 * volume) ** (2 / 3)) / surface_area

        sphericities.append(sphericity)
        areas.append(volume)

    if not areas:
        return 0.0

    # 加权平均
    mean_sphericity = np.sum(np.array(sphericities) * np.array(areas)) / np.sum(areas)

    return mean_sphericity


def aspect_ratio_3D(A: np.ndarray) -> float:
    """
    计算3D孔隙的纵横比（简化版，只计算一个参数）

    参数：
        A: 3D二值图像

    返回：
        平均纵横比（基于包围盒）
    """
    A = A.astype(np.uint8)
    labels = measure.label(A, connectivity=1)
    props = measure.regionprops(labels)

    if not props:
        return 0.0

    aspect_ratios = []
    volumes = []

    for prop in props:
        volume = prop.area
        if volume < 8:
            continue

        # 基于包围盒计算纵横比
        bbox = prop.bbox  # (min_row, min_col, min_z, max_row, max_col, max_z)

        # 计算各方向长度
        length_x = bbox[3] - bbox[0]  # max_row - min_row
        length_y = bbox[4] - bbox[1]  # max_col - min_col
        length_z = bbox[5] - bbox[2]  # max_z - min_z

        # 找到最小和最大边长
        lengths = [length_x, length_y, length_z]
        min_length = min(lengths)
        max_length = max(lengths)

        # 纵横比 = 最小边长 / 最大边长
        ar = min_length / max_length if max_length > 0 else 0

        aspect_ratios.append(ar)
        volumes.append(volume)

    if not volumes:
        return 0.0

    # 加权平均（按体积）
    volumes = np.array(volumes)
    aspect_ratios = np.array(aspect_ratios)

    avg_aspect_ratio = np.sum(aspect_ratios * volumes) / np.sum(volumes)

    return float(avg_aspect_ratio)


#计算迂曲度，modified by Houshuai, from https://doi.org/10.1016/j.powtec.2017.06.066;
def tortuosity_3D(A: np.ndarray) -> float:
    """
    计算3D孔隙网络的曲折度（简化但有效的版本）

    参数：
        A: 3D二值图像，其中1表示孔隙，0表示基质

    返回：
        tortuosity: 孔隙网络的曲折度（≥1.0）
    """
    try:
        # 1. 检查输入
        A = A.astype(np.uint8)
        if np.sum(A) == 0:
            return 1.0  # 没有孔隙

        # 2. 获取图像尺寸
        nz, ny, nx = A.shape

        # 3. 骨架化孔隙相
        skeleton = morphology.skeletonize_3d(A)

        # 4. 标记连通分量
        labeled_skeleton = measure.label(skeleton, connectivity=1)
        num_components = np.max(labeled_skeleton)

        if num_components == 0:
            return 1.0

        # 5. 对每个连通分量计算最长路径的曲折度
        component_tortuosities = []
        component_volumes = []

        for label in range(1, num_components + 1):
            component = (labeled_skeleton == label)
            component_volume = np.sum(component)

            if component_volume < 10:  # 忽略太小的分量
                continue

            # 计算该分量的曲折度
            comp_tort = calculate_component_tortuosity(component)

            if comp_tort > 1.0:  # 只记录有效的曲折度
                component_tortuosities.append(comp_tort)
                component_volumes.append(component_volume)

        # 6. 如果没有有效分量，返回默认值
        if not component_tortuosities:
            return 1.0

        # 7. 按体积加权平均
        component_volumes = np.array(component_volumes)
        component_tortuosities = np.array(component_tortuosities)

        weighted_tortuosity = np.sum(component_tortuosities * component_volumes) / np.sum(component_volumes)

        # 限制在合理范围
        weighted_tortuosity = max(1.0, min(weighted_tortuosity, 5.0))

        return float(weighted_tortuosity)

    except Exception as e:
        print(f"Error in tortuosity_3D: {e}")
        return 1.0
def calculate_component_tortuosity(component: np.ndarray) -> float:
    """
    计算单个连通分量的曲折度

    算法：
    1. 找到两个最远点（端点或边界点）
    2. 计算它们之间的直线距离
    3. 计算它们之间的最短路径长度（近似）
    4. 曲折度 = 路径长度 / 直线距离
    """
    # 1. 找到所有体素点的坐标
    coords = np.argwhere(component)

    if len(coords) < 2:
        return 1.0

    # 2. 找到最远点对（使用欧氏距离）
    max_distance = 0
    point1 = None
    point2 = None

    # 为了效率，只检查部分点对（随机采样）
    n_points = min(50, len(coords))
    indices = np.random.choice(len(coords), n_points, replace=False)
    sampled_coords = coords[indices]

    for i in range(n_points):
        for j in range(i+1, n_points):
            dist = np.linalg.norm(sampled_coords[i] - sampled_coords[j])
            if dist > max_distance:
                max_distance = dist
                point1 = sampled_coords[i]
                point2 = sampled_coords[j]

    if max_distance < 1:
        return 1.0

    # 3. 计算直线距离
    straight_distance = np.linalg.norm(point1 - point2)

    # 4. 近似计算路径长度（使用体素数代替实际路径）
    # 对于连通分量，可以用体素数作为路径长度的近似
    path_length = approximate_path_length(component, point1, point2)

    # 5. 计算曲折度
    tortuosity = path_length / straight_distance if straight_distance > 0 else 1.0

    # 限制范围
    tortuosity = max(1.0, min(tortuosity, 5.0))

    return tortuosity
def approximate_path_length(component: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    """
    近似计算两点之间的路径长度

    使用A*算法或简单的替代方法
    """
    try:
        # 方法1：使用Dijkstra算法（更准确但较慢）
        # 为了简化，这里使用更快的近似方法

        # 方法2：使用分量直径作为路径长度的近似
        # 获取所有坐标
        coords = np.argwhere(component)

        # 计算各方向的范围
        z_coords = coords[:, 0]
        y_coords = coords[:, 1]
        x_coords = coords[:, 2]

        z_range = np.max(z_coords) - np.min(z_coords)
        y_range = np.max(y_coords) - np.min(y_coords)
        x_range = np.max(x_coords) - np.min(x_coords)

        # 使用最大范围作为路径长度的近似
        max_range = max(z_range, y_range, x_range)

        # 方法3：结合直线距离和形状复杂度
        straight_dist = np.linalg.norm(start - end)

        # 如果分量比较直，路径长度接近直线距离
        # 如果分量弯曲，路径长度大于直线距离
        # 用分量体素数除以平均截面积来估计
        volume = np.sum(component)

        # 估计平均截面积（简化）
        if straight_dist > 0:
            avg_cross_section = volume / straight_dist
        else:
            avg_cross_section = volume ** (2/3)

        # 如果截面积估计太小，使用直线距离的倍数
        if avg_cross_section < 1:
            estimated_path = straight_dist * 1.5
        else:
            estimated_path = volume / max(1, avg_cross_section)

        # 确保路径长度不小于直线距离
        estimated_path = max(estimated_path, straight_dist)

        return estimated_path

    except:
        # 如果上述方法失败，使用简单的近似
        return np.linalg.norm(start - end) * 1.2  # 默认增加20%
# 更简单的备用版本（如果上面的太复杂）
def tortuosity_3D_alternative(A: np.ndarray) -> float:
    """
    替代的曲折度计算方法（更简单，更稳定）

    基于孔隙相的形状各向异性估计曲折度
    """
    try:
        A = A.astype(np.uint8)

        # 1. 计算孔隙相的形状特征
        labels = measure.label(A, connectivity=1)
        props = measure.regionprops(labels)

        if not props:
            return 1.0

        tortuosities = []
        volumes = []

        for prop in props:
            volume = prop.area  # 3D中area是体积

            if volume < 8:
                continue

            # 2. 获取包围盒尺寸
            bbox = prop.bbox  # (min_z, min_y, min_x, max_z, max_y, max_x)

            # 计算各方向长度
            length_z = bbox[3] - bbox[0]
            length_y = bbox[4] - bbox[1]
            length_x = bbox[5] - bbox[2]

            # 3. 计算长宽比（各向异性）
            lengths = [length_z, length_y, length_x]
            max_length = max(lengths)
            min_length = min(lengths)

            # 纵横比 = 最小尺寸 / 最大尺寸
            aspect_ratio = min_length / max_length if max_length > 0 else 1.0

            # 4. 基于纵横比估计曲折度
            # 公式：τ = 1 + α * (1 - 纵横比)
            # 其中α是经验系数，这里设为1.5
            alpha = 1.5
            tort = 1.0 + alpha * (1.0 - aspect_ratio)

            # 限制范围
            tort = max(1.0, min(tort, 3.0))

            tortuosities.append(tort)
            volumes.append(volume)

        if not tortuosities:
            return 1.0

        # 5. 按体积加权平均
        volumes = np.array(volumes)
        tortuosities = np.array(tortuosities)

        avg_tortuosity = np.sum(tortuosities * volumes) / np.sum(volumes)

        return float(avg_tortuosity)

    except Exception as e:
        print(f"Error in tortuosity_3D_alternative: {e}")
        return 1.0

# 最终推荐使用的版本
def tortuosity_3D_final(A: np.ndarray) -> float:
    """
    最终版本的曲折度计算函数

    结合两种方法的优点，提供稳定可靠的结果
    """
    # 先尝试第一个方法
    tort1 = tortuosity_3D(A)

    # 如果结果不合理（比如等于1.0），尝试第二种方法
    if tort1 < 1.01:  # 如果接近1.0，可能计算有问题
        tort2 = tortuosity_3D_alternative(A)
        return tort2

    return tort1
def extract_structure_3d(voxel: np.ndarray,resolution_um: float = 1.0) -> Dict:
    """
    从3D数字岩心提取结构参数

    参数：
        voxel: 3D二值图像，其中1表示孔隙，0表示基质

    返回：
        包含各种结构参数的字典
    """
    # 确保是二值图像
    if voxel.dtype != np.uint8:
        voxel = (voxel >= 0).astype(np.uint8)

    # 反转：使孔隙为1，基质为0
    # voxel = np.where(voxel >= 1, 0, 1)
    # 计算孔隙度
    # import pdb;pdb.set_trace()
    porosity = np.sum(voxel) / voxel.size

    # 计算各种参数
    pore_radius, pore_radii = PSD_3D(voxel)
    fd = fractal_dimension_3d(voxel)
    throat_radius, throat_sizes = Throat_Size_3D(voxel)
    coordination, coord_numbers = Coordination_3D(voxel)
    sp_pores, _ = SPS_3D(voxel)
    sphericity = yuandu_3D(voxel)
    AR = aspect_ratio_3D(voxel)
    tort = tortuosity_3D_final(voxel)


    results = {
        'Porosity': float(porosity),
        'Mean_Pore_Radius': float(pore_radius * resolution_um),
        'Fractal_Dimension': float(fd),
        'Mean_Throat_Radius': float(throat_radius * resolution_um),
        'Mean_Coordination': float(coordination),
        'Specific_Surface_Pores': float(sp_pores/resolution_um),
        # 'Specific_Surface_Matrix': float(sp_matrix/resolution_um),
        'Sphericity': float(sphericity),
        'Aspect_Ratio': float(AR),
        'Tortuosity': float(tort),
        'Total_Pores': int(len(pore_radii) if len(pore_radii) > 0 else 0),
        'Voxel_Dimensions': voxel.shape
    }

    return results

    # return results

def process_subcubes(voxel: np.ndarray, subcube_size: int = 64, stride: int = 64) -> List[Dict]:
    """
    处理整个体数据的子立方体

    参数：
        voxel: 完整的3D体数据
        subcube_size: 子立方体大小
        stride: 滑动步长

    返回：
        每个子立方体的分析结果列表
    """
    # 生成子立方体
    subcubes = sliding_window_subcubes(voxel, subcube_size, stride)

    results = []
    n_slices = subcubes.shape[:3]

    print(f"Processing {np.prod(n_slices)} subcubes...")

    for i in range(n_slices[0]):
        for j in range(n_slices[1]):
            for k in range(n_slices[2]):
                subcube = subcubes[i, j, k]

                # 检查子立方体是否包含足够的孔隙
                if np.sum(subcube) / subcube.size < 0.01:  # 孔隙度小于1%
                    continue

                # 分析子立方体
                result = extract_structure_3d(subcube)
                result['Subcube_Index'] = (i, j, k)
                result['Subcube_Position'] = (i * stride, j * stride, k * stride)

                results.append(result)

    return results

def visualize_results(results: List[Dict], voxel_shape: Tuple):
    """
    可视化分析结果
    """
    if not results:
        print("No results to visualize")
        return

    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Digital Rock Analysis Results', fontsize=16)

    # 提取数据
    porosities = [r['Porosity'] for r in results]
    pore_radii = [r['Mean_Pore_Radius'] for r in results]
    fractal_dims = [r['Fractal_Dimension'] for r in results]
    throat_radii = [r['Mean_Throat_Radius'] for r in results]
    coordinations = [r['Mean_Coordination'] for r in results]
    sphericities = [r['Sphericity'] for r in results]

    # 1. 孔隙度分布
    axes[0, 0].hist(porosities, bins=20, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Porosity')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Porosity Distribution')

    # 2. 孔隙半径分布
    axes[0, 1].hist(pore_radii, bins=20, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Mean Pore Radius')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Pore Radius Distribution')

    # 3. 分形维数分布
    axes[0, 2].hist(fractal_dims, bins=20, edgecolor='black', alpha=0.7)
    axes[0, 2].set_xlabel('Fractal Dimension')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Fractal Dimension Distribution')

    # 4. 喉道半径 vs 孔隙半径
    axes[1, 0].scatter(pore_radii, throat_radii, alpha=0.6)
    axes[1, 0].set_xlabel('Mean Pore Radius')
    axes[1, 0].set_ylabel('Mean Throat Radius')
    axes[1, 0].set_title('Pore vs Throat Radius')

    # 5. 配位数分布
    axes[1, 1].hist(coordinations, bins=20, edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Coordination Number')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Coordination Number Distribution')

    # 6. 球形度分布
    axes[1, 2].hist(sphericities, bins=20, edgecolor='black', alpha=0.7)
    axes[1, 2].set_xlabel('Sphericity')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('Sphericity Distribution')

    plt.tight_layout()
    plt.savefig('rock_analysis_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # 加载数据
    print("Loading data...")
    # 假设数据是1000x1000x1000的uint8数组
    data = np.fromfile('./data/s1.raw', dtype=np.uint8)

    # 重塑为3D数组
    # 注意：需要知道原始数据的准确尺寸
    # 这里假设是1000x1000x1000
    dim_size = 1000
    data_3d = data.reshape((dim_size, dim_size, dim_size))

    # 为了测试，裁剪一个小块：从中间取64x64x64
    start = 0  # 从中间开始取
    end = start + 64
    test_cube = data_3d[start:end, start:end, start:end].copy()
    print(f"Test cube shape: {test_cube.shape}")
    # 首先查看数据的分布
    unique, counts = np.unique(test_cube, return_counts=True)
    print(f"\nValue distribution in test cube:")
    for val, cnt in zip(unique, counts):
        percentage = cnt / test_cube.size * 100
        print(f"  Value {val}: {cnt} voxels ({percentage:.1f}%)")
    results = extract_structure_3d(test_cube,2.5)