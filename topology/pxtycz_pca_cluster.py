import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.spatial.distance import cdist
import seaborn as sns
import logging
import os
import pickle
import json

# 设置中文字体为微软雅黑
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(level=logging.INFO, format="%(message)s")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.spatial.distance import cdist
import seaborn as sns
import logging
import os
import pickle
import json

# 设置中文字体为微软雅黑
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(level=logging.INFO, format="%(message)s")


class PoreTopologyModel:
    """孔隙拓扑模型保存和加载类"""

    def __init__(self):
        self.pca_weights = {}
        self.scaler_params = {}
        # 核心修改：用于泛化归一化和 PxTyCz 分类的参数
        self.normalization_params = {}
        self.pca_components = {}
        self.model_info = {}

    def save_from_pca_info(self, pca_info, scalers, output_path, normalization_params):
        """
        从PCA信息保存模型参数 (新增 normalization_params 参数)

        参数:
        - pca_info: PCA分析结果
        - scalers: 标准化器
        - output_path: 输出路径
        - normalization_params: 固化的归一化和分类边界参数 (新增加的)
        """
        # 1. 保存PCA权重和主成分
        self.pca_weights = {
            'connectivity': {
                'components': pca_info['coord_components'].tolist(),
                'explained_variance': float(pca_info['explained_variances'][0]),
                'feature_names': ['prop_isolated', 'prop_dead_end', 'prop_well_connected', 'prop_highly_connected']
            },
            'pore_size': {
                'components': pca_info['pore_components'].tolist(),
                'explained_variance': float(pca_info['explained_variances'][1]),
                'feature_names': ['prop_nanopore', 'prop_micropore', 'prop_mesopore', 'prop_macropore']
            },
            'throat_size': {
                'components': pca_info['throat_components'].tolist(),
                'explained_variance': float(pca_info['explained_variances'][2]),
                'feature_names': ['prop_nano_throat_dominant', 'prop_micro_throat_dominant',
                                  'prop_meso_throat_dominant', 'prop_macro_throat_dominant']
            }
        }

        # 2. 保存标准化器参数
        self.scaler_params = {
            'connectivity': {
                'mean': scalers['connectivity'].mean_.tolist() if hasattr(scalers['connectivity'], 'mean_') else [],
                'scale': scalers['connectivity'].scale_.tolist() if hasattr(scalers['connectivity'], 'scale_') else []
            },
            'pore_size': {
                'mean': scalers['pore_size'].mean_.tolist() if hasattr(scalers['pore_size'], 'mean_') else [],
                'scale': scalers['pore_size'].scale_.tolist() if hasattr(scalers['pore_size'], 'scale_') else []
            },
            'throat_size': {
                'mean': scalers['throat_size'].mean_.tolist() if hasattr(scalers['throat_size'], 'mean_') else [],
                'scale': scalers['throat_size'].scale_.tolist() if hasattr(scalers['throat_size'], 'scale_') else []
            }
        }

        # 3. 核心修改：保存固化的归一化和分类参数
        self.normalization_params = normalization_params

        # 4. 保存模型信息
        self.model_info = {
            'creation_date': pd.Timestamp.now().isoformat(),
            'version': '1.0',
            'description': 'Pore Topology Classification Model based on PCA',
            'n_components': 1,
            'random_state': 42,
            'method': 'PCA-based ternary coordinates with PxTyCz classification'  # 更新描述
        }

        # 5. 保存到文件
        self._save_all_formats(output_path)

        return self

    def _save_all_formats(self, output_path):
        """保存所有格式的文件"""
        os.makedirs(output_path, exist_ok=True)

        # 1. 保存为Python pickle文件（完整模型）
        pickle_file = os.path.join(output_path, 'pore_topology_model.pkl')
        with open(pickle_file, 'wb') as f:
            pickle.dump({
                'pca_weights': self.pca_weights,
                'scaler_params': self.scaler_params,
                'normalization_params': self.normalization_params,  # 新增
                'model_info': self.model_info
            }, f)
        logging.info(f"模型已保存为pickle文件: {pickle_file}")

        # 2. 保存为JSON文件（人类可读）
        json_file = os.path.join(output_path, 'pore_topology_model.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                'pca_weights': self.pca_weights,
                'scaler_params': self.scaler_params,
                'normalization_params': self.normalization_params,  # 新增
                'model_info': self.model_info
            }, f, indent=2, ensure_ascii=False)
        logging.info(f"模型已保存为JSON文件: {json_file}")

        # 3. 保存为TXT文件（简单格式）
        txt_file = os.path.join(output_path, 'pore_topology_weights.txt')
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("孔隙拓扑模型权重参数\n")
            f.write("=" * 60 + "\n\n")

            # 写入PCA和StandardScaler参数
            for dim_name, weights in self.pca_weights.items():
                f.write(f"=== {dim_name.upper()} 维度 ===\n")
                f.write(f"解释方差: {weights['explained_variance']:.4f}\n")
                f.write("特征权重:\n")

                for i, (feature, weight) in enumerate(zip(weights['feature_names'], weights['components'])):
                    f.write(f"  {feature}: {weight:.6f}\n")

                # 标准化器参数
                if dim_name in self.scaler_params:
                    f.write(f"\n标准化器参数:\n")
                    f.write(f"  均值: {self.scaler_params[dim_name]['mean']}\n")
                    f.write(f"  标准差: {self.scaler_params[dim_name]['scale']}\n")

                f.write("\n")

            # 写入固化的归一化和分类参数
            f.write("=" * 60 + "\n")
            f.write("通用归一化和分类参数\n")
            f.write("=" * 60 + "\n")
            if self.normalization_params:
                f.write(f"PCA得分最小值 (fc1, fc2, fc3): {self.normalization_params['pca_min_values']}\n")
                f.write(f"PCA得分最大值 (拉伸后): {self.normalization_params['pca_max_values']}\n")
                f.write(f"分类区间数 (N): {self.normalization_params['n_intervals']}\n")
                f.write("分类边界 (P/T/C的25%/50%/75%分位数):\n")
                for k, v in self.normalization_params['classification_boundaries'].items():
                    f.write(f"  {k}: {v}\n")
            else:
                f.write("未保存通用归一化和分类参数。\n")

            f.write("\n")
            f.write("=" * 60 + "\n")
            f.write("模型信息\n")
            f.write("=" * 60 + "\n")
            for key, value in self.model_info.items():
                f.write(f"{key}: {value}\n")

        logging.info(f"模型权重已保存为TXT文件: {txt_file}")

        # 4. 保存为CSV文件（表格格式，仅保存权重，保持原样）
        csv_file = os.path.join(output_path, 'pore_topology_weights.csv')
        csv_data = []

        for dim_name, weights in self.pca_weights.items():
            for feature, weight in zip(weights['feature_names'], weights['components']):
                csv_data.append({
                    'dimension': dim_name,
                    'feature': feature,
                    'weight': weight,
                    'explained_variance': weights['explained_variance']
                })

        pd.DataFrame(csv_data).to_csv(csv_file, index=False)
        logging.info(f"模型权重已保存为CSV文件: {csv_file}")

    def load_model(self, model_path):
        """加载模型"""
        if model_path.endswith('.pkl'):
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
        elif model_path.endswith('.json'):
            with open(model_path, 'r', encoding='utf-8') as f:
                model_data = json.load(f)
        else:
            raise ValueError("不支持的模型格式，请使用.pkl或.json文件")

        self.pca_weights = model_data['pca_weights']
        self.scaler_params = model_data.get('scaler_params', {})
        self.normalization_params = model_data.get('normalization_params', {})  # 核心修改：加载固化参数
        self.model_info = model_data.get('model_info', {})

        logging.info(f"模型已从 {model_path} 加载")
        logging.info(f"模型创建时间: {self.model_info.get('creation_date', '未知')}")

        # 验证是否包含通用归一化参数
        if not self.normalization_params:
            logging.warning(
                "模型文件缺少通用归一化和分类参数（normalization_params），新数据投影可能不稳定且无法进行 PxTyCz 分类。")

        return self

    def apply_to_new_data(self, df):
        """
        将模型应用到新数据，使用固化的统计量进行标准化、三元坐标转换和 PxTyCz 分类。
        【修复版本：增加数据截断机制】
        """
        if not self.pca_weights:
            raise ValueError("请先加载或训练模型")

        logging.info("=== 应用模型到新数据（修复版） ===")

        # 提取特征数据
        features_by_dimension = {
            'connectivity': ['prop_isolated', 'prop_dead_end', 'prop_well_connected', 'prop_highly_connected'],
            'pore_size': ['prop_nanopore', 'prop_micropore', 'prop_mesopore', 'prop_macropore'],
            'throat_size': ['prop_nano_throat_dominant', 'prop_micro_throat_dominant',
                            'prop_meso_throat_dominant', 'prop_macro_throat_dominant']
        }

        dim_scores = []

        for dim_name in ['connectivity', 'pore_size', 'throat_size']:
            features = features_by_dimension[dim_name]

            weights = np.array(self.pca_weights[dim_name]['components']).flatten()
            data = df[features].values

            # 应用固化的 StandardScaler 参数
            if dim_name in self.scaler_params and self.scaler_params[dim_name].get('mean'):
                mean = np.array(self.scaler_params[dim_name]['mean'])
                scale = np.array(self.scaler_params[dim_name]['scale'])
                data_scaled = (data - mean) / scale
            else:
                logging.warning(f"维度 {dim_name}: 使用未标准化的数据")
                data_scaled = data

            # 应用固化的 PCA 权重进行投影
            pca_score = np.dot(data_scaled, weights)
            dim_scores.append(pca_score.flatten())

            # 调试信息
            logging.debug(f"维度 {dim_name}: PCA得分范围 [{np.min(pca_score):.2f}, {np.max(pca_score):.2f}]")

        # 组合成 PCA 原始得分（包含负值）
        ternary_coords_raw = np.column_stack(dim_scores)

        logging.info(f"原始PCA得分统计:")
        logging.info(
            f"  FC1 (连通性): min={np.min(ternary_coords_raw[:, 0]):.4f}, max={np.max(ternary_coords_raw[:, 0]):.4f}")
        logging.info(
            f"  FC2 (孔隙尺寸): min={np.min(ternary_coords_raw[:, 1]):.4f}, max={np.max(ternary_coords_raw[:, 1]):.4f}")
        logging.info(
            f"  FC3 (喉道尺寸): min={np.min(ternary_coords_raw[:, 2]):.4f}, max={np.max(ternary_coords_raw[:, 2]):.4f}")

        # *************************************************************
        # 核心步骤 3: 使用固化的 Min/Max 参数进行三元坐标转换（修复鲁棒性）
        # *************************************************************
        if not self.normalization_params or 'pca_min_values' not in self.normalization_params:
            logging.error("缺少固化的 Min/Max 归一化参数，无法进行通用投影。")
            # 回退逻辑
            min_val = np.min(ternary_coords_raw, axis=0)
            max_val = np.max(ternary_coords_raw - min_val + 1e-8, axis=0)
            shift_constant = 1e-8
        else:
            # 使用固化参数
            min_val = np.array(self.normalization_params['pca_min_values'])
            max_val = np.array(self.normalization_params['pca_max_values'])
            shift_constant = self.normalization_params['shift_constant']

            logging.info("使用固化归一化参数:")
            logging.info(f"  固化min值: [{min_val[0]:.4f}, {min_val[1]:.4f}, {min_val[2]:.4f}]")
            logging.info(f"  固化max值: [{max_val[0]:.4f}, {max_val[1]:.4f}, {max_val[2]:.4f}]")
            logging.info(f"  固化shift常数: {shift_constant}")

        # === 关键修正：增加数据截断机制 ===

        # 1. 平移数据 (基于训练集的最小值)
        shifted_data = ternary_coords_raw - min_val + shift_constant

        # 记录截断前的统计
        logging.debug("截断前shifted_data统计:")
        for i, dim in enumerate(['FC1', 'FC2', 'FC3']):
            neg_count = np.sum(shifted_data[:, i] < 0)
            neg_percent = neg_count / len(shifted_data) * 100
            if neg_count > 0:
                logging.warning(f"  {dim}: 有 {neg_count} 个负值 ({neg_percent:.1f}%)")

        # 2. 【关键修正】截断负值 (Clip Negative Values)
        # 如果新数据比训练集最小值还小，shifted_data 会是负数，这会导致三元图错误。
        # 物理含义：该维度的特征极弱（比训练集见过的最弱还弱），我们将其视为 0。
        shifted_data = np.maximum(shifted_data, 0)  # 强制非负

        # 记录截断后的统计
        logging.info("截断后shifted_data统计:")
        for i, dim in enumerate(['FC1', 'FC2', 'FC3']):
            logging.info(f"  {dim}: [{np.min(shifted_data[:, i]):.4f}, {np.max(shifted_data[:, i]):.4f}]")

        # 3. 处理上界溢出（可选，增加鲁棒性）
        # 允许稍微超出训练集最大值，但不无限大
        # 例如：限制在训练集最大值的2倍范围内
        max_val_clipped = max_val * 2.0
        shifted_data = np.minimum(shifted_data, max_val_clipped)

        # 4. 归一化到参考范围
        normalized_coords = shifted_data / max_val

        # 记录归一化后统计
        logging.info("归一化后坐标统计:")
        for i, dim in enumerate(['FC1', 'FC2', 'FC3']):
            logging.info(f"  {dim}: [{np.min(normalized_coords[:, i]):.4f}, {np.max(normalized_coords[:, i]):.4f}]")

        # 5. 确保三元坐标和为1
        row_sums = np.sum(normalized_coords, axis=1, keepdims=True)

        # 处理特殊情况：防止全0行（即三个维度都极小，低于训练集最小值）
        zero_rows = np.sum(row_sums == 0)
        if zero_rows > 0:
            logging.warning(f"发现 {zero_rows} 个全0行（极端异常点），分配平均坐标")
            # 对于全0行，分配平均坐标 [0.333, 0.333, 0.333]
            avg_coords = np.ones_like(normalized_coords) / 3.0
            normalized_coords[row_sums.flatten() == 0] = avg_coords[0]
            row_sums = np.sum(normalized_coords, axis=1, keepdims=True)

        # 防止除零
        row_sums[row_sums == 0] = 1e-8

        final_coords = normalized_coords / row_sums

        # 验证最终坐标的合法性
        logging.info("最终三元坐标验证:")
        for i, dim in enumerate(['FC1', 'FC2', 'FC3']):
            min_val = np.min(final_coords[:, i])
            max_val = np.max(final_coords[:, i])
            status = "✓ OK" if min_val >= 0 and max_val <= 1 else "✗ ERROR"
            logging.info(f"  {dim}: [{min_val:.4f}, {max_val:.4f}] {status}")

        # 检查坐标和
        sums = np.sum(final_coords, axis=1)
        min_sum = np.min(sums)
        max_sum = np.max(sums)
        sum_status = "✓ OK" if np.allclose(sums, 1.0, rtol=1e-5) else f"✗ ERROR ({min_sum:.4f}-{max_sum:.4f})"
        logging.info(f"坐标和: {sum_status}")

        # 统计异常点
        invalid_points = 0
        for i in range(len(final_coords)):
            if not (np.all(final_coords[i] >= 0) and np.all(final_coords[i] <= 1) and
                    np.abs(np.sum(final_coords[i]) - 1.0) < 0.01):
                invalid_points += 1
                if invalid_points <= 3:  # 只显示前3个异常点
                    logging.warning(f"异常点 {i} ({df.index[i]}): {final_coords[i]}, sum={np.sum(final_coords[i]):.4f}")

        if invalid_points > 0:
            logging.warning(f"总计有 {invalid_points} 个异常点 ({invalid_points / len(final_coords) * 100:.1f}%)")
        else:
            logging.info("所有坐标点都在有效范围内")

        # *************************************************************
        # 核心步骤 4: 分配 PxTyCz 拓扑类型
        # *************************************************************
        categories = self._assign_topology_category(final_coords)

        logging.info(f"新数据转换完成: {len(final_coords)}个样本")
        logging.info(f"新数据的拓扑分类 (前3个): {categories[:3]}")

        # 统计分类分布
        category_counts = {}
        for cat in categories:
            category_counts[cat] = category_counts.get(cat, 0) + 1

        logging.info("拓扑分类分布:")
        for cat, count in sorted(category_counts.items()):
            percentage = count / len(categories) * 100
            logging.info(f"  {cat}: {count}个 ({percentage:.1f}%)")

        return final_coords, categories

    def _assign_topology_category(self, final_coords):
        """使用固化分类边界进行 PxTyCz 命名"""
        if not self.normalization_params.get('classification_boundaries'):
            logging.warning("缺少固化的分类边界参数，无法生成 PxTyCz 标签。")
            return ['N/A'] * len(final_coords)

        boundaries = self.normalization_params['classification_boundaries']
        n_intervals = self.normalization_params['n_intervals']

        categories = []

        # 定义维度名称和对应的坐标索引
        # fc1: Connectivity (C), fc2: Pore Size (P), fc3: Throat Size (T)
        dim_map = {
            'fc1_connectivity': (0, 'c'),
            'fc2_pore_size': (1, 'p'),
            'fc3_throat_size': (2, 't')
        }

        # 为了更好地理解 PxTyCz 的概念，可以参考三元图的划分
        #

        for i in range(len(final_coords)):
            label = {}
            for dim_name, (idx, prefix) in dim_map.items():
                value = final_coords[i, idx]
                # 默认是最大区间 N
                interval_id = n_intervals

                # 遍历固化边界，确定区间 (等频划分)
                # 分位数是 25%, 50%, 75%
                # 边界[b1, b2, b3] -> <b1=p1, <b2=p2, <b3=p3, >b3=p4
                for k, boundary in enumerate(boundaries[dim_name]):
                    if value <= boundary:
                        interval_id = k + 1
                        break

                label[prefix] = str(interval_id)

            # 组合成 PxTyCz 标签
            categories.append(f"p{label['p']}t{label['t']}c{label['c']}")

        return categories

def calculate_ternary_coordinates_pca(coord_data, pore_data, throat_data, save_model=False, output_path=None):
    """
    使用PCA方法计算三元坐标 - 优化版本，增加模型保存功能和泛化归一化参数。
    """
    logging.info("=== 使用PCA方法进行维度融合 ===")

    # 对每个维度分别进行PCA
    # 注意：这里的 scaler 必须是独立的实例，确保 fit_transform 是针对各自维度进行的
    pca_conn = PCA(n_components=1, random_state=42)
    scaler_conn = StandardScaler()

    pca_pore = PCA(n_components=1, random_state=42)
    scaler_pore = StandardScaler()

    pca_throat = PCA(n_components=1, random_state=42)
    scaler_throat = StandardScaler()

    # 连通性维度PCA (fc1)
    coord_scaled = scaler_conn.fit_transform(coord_data)  # 使用独立的 scaler_conn
    coord_pca = pca_conn.fit_transform(coord_scaled)
    coord_variance = pca_conn.explained_variance_ratio_[0]
    coord_components = pca_conn.components_[0]

    # 孔隙尺寸维度PCA (fc2)
    pore_scaled = scaler_pore.fit_transform(pore_data)  # 使用独立的 scaler_pore
    pore_pca = pca_pore.fit_transform(pore_scaled)
    pore_variance = pca_pore.explained_variance_ratio_[0]
    pore_components = pca_pore.components_[0]

    # 喉道尺寸维度PCA (fc3)
    throat_scaled = scaler_throat.fit_transform(throat_data)  # 使用独立的 scaler_throat
    throat_pca = pca_throat.fit_transform(throat_scaled)
    throat_variance = pca_throat.explained_variance_ratio_[0]
    throat_components = pca_throat.components_[0]

    # 打印PCA分析结果 (保持不变)
    logging.info("PCA解释方差比:")
    logging.info(f"  连通性: {coord_variance:.3f}")
    logging.info(f"  孔隙尺寸: {pore_variance:.3f}")
    logging.info(f"  喉道尺寸: {throat_variance:.3f}")

    logging.info("PCA主成分权重:")
    coord_features = ['孤立', '死端', '良好连通', '高度连通']
    pore_features = ['纳米孔隙', '微米孔隙', '中孔', '宏观孔隙']
    throat_features = ['纳米喉道', '微米喉道', '中孔喉道', '宏观喉道']

    logging.info("连通性主成分:")
    for i, (name, weight) in enumerate(zip(coord_features, coord_components)):
        logging.info(f"  {name}: {weight:.3f}")

    logging.info("孔隙尺寸主成分:")
    for i, (name, weight) in enumerate(zip(pore_features, pore_components)):
        logging.info(f"  {name}: {weight:.3f}")

    logging.info("喉道尺寸主成分:")
    for i, (name, weight) in enumerate(zip(throat_features, throat_components)):
        logging.info(f"  {name}: {weight:.3f}")

    # 组合PCA结果 (PCA原始得分，可能包含负值)
    ternary_coords_raw = np.column_stack([
        coord_pca.flatten(),
        pore_pca.flatten(),
        throat_pca.flatten()
    ])

    # **********************************************
    # 核心修改 1: 固化归一化参数（用于泛化投影）
    # **********************************************

    # 1. 固化 Min 值：将所有 PCA 得分都拉到正值范围的最小值
    min_val_fixed = np.min(ternary_coords_raw, axis=0)

    # 2. 固化 Max 值：计算拉伸后的最大值
    shift_constant = 1e-8
    shifted_data = ternary_coords_raw - min_val_fixed + shift_constant
    max_val_fixed = np.max(shifted_data, axis=0)

    # 3. 应用固化的 Min/Max 进行归一化 (训练集数据使用自身固化的边界)
    normalized_coords = shifted_data / max_val_fixed

    # 确保三元坐标和为1
    row_sums = np.sum(normalized_coords, axis=1, keepdims=True)
    # 避免除零
    row_sums[row_sums == 0] = 1
    final_coords = normalized_coords / row_sums

    # 打印最终坐标统计 (保持不变)
    logging.info("最终三元坐标统计:")
    logging.info(f"  连通性: [{final_coords[:, 0].min():.3f}, {final_coords[:, 0].max():.3f}]")
    logging.info(f"  孔隙尺寸: [{final_coords[:, 1].min():.3f}, {final_coords[:, 1].max():.3f}]")
    logging.info(f"  喉道尺寸: [{final_coords[:, 2].min():.3f}, {final_coords[:, 2].max():.3f}]")

    # 准备PCA信息
    pca_info = {
        'coord_components': coord_components,
        'pore_components': pore_components,
        'throat_components': throat_components,
        'explained_variances': [coord_variance, pore_variance, throat_variance]
    }

    # 准备标准化器（独立实例）
    scalers = {
        'connectivity': scaler_conn,
        'pore_size': scaler_pore,
        'throat_size': scaler_throat
    }

    # 如果需要保存模型
    if save_model and output_path:
        # **********************************************
        # 核心修改 2: 计算并保存 PxTyCz 分类边界
        # **********************************************
        n_intervals = 4  # P1到P4
        # 等频划分的分位数：25%, 50%, 75%
        quantiles = np.linspace(0.25, 0.75, n_intervals - 1)

        classification_boundaries = {
            # fc1: 连通性
            'fc1_connectivity': np.quantile(final_coords[:, 0], quantiles).tolist(),
            # fc2: 孔隙尺寸
            'fc2_pore_size': np.quantile(final_coords[:, 1], quantiles).tolist(),
            # fc3: 喉道尺寸
            'fc3_throat_size': np.quantile(final_coords[:, 2], quantiles).tolist()
        }

        # 封装所有需要固化的参数
        normalization_params = {
            'pca_min_values': min_val_fixed.tolist(),  # 固化的 Min 值
            'pca_max_values': max_val_fixed.tolist(),  # 固化的 Max 值
            'shift_constant': shift_constant,
            'classification_boundaries': classification_boundaries,  # 固化的分类边界
            'n_intervals': n_intervals
        }

        # 调用模型保存器，传入所有固化参数
        model_saver = PoreTopologyModel()
        model_saver.save_from_pca_info(pca_info, scalers, output_path, normalization_params)

        logging.info("模型和所有泛化参数（包括P/T/C分类边界）已保存。")

    return final_coords, pca_info, scalers

def create_composite_features_pca(coord_data, pore_data, throat_data, save_model=False, output_path=None):
    """
    使用PCA方法创建特征 - 统一使用三元坐标，增加模型保存选项
    """
    # 计算三元坐标
    weighted_features, pca_info, scalers = calculate_ternary_coordinates_pca(
        coord_data, pore_data, throat_data, save_model=save_model, output_path=output_path
    )

    # 聚类和可视化都使用相同的三元坐标
    return weighted_features, weighted_features, pca_info
def save_cluster_mapping_with_coordinates(results_df, weighted_features, output_path):
    """
    保存样本名称、聚类类别和三元坐标
    """
    cluster_mapping_file = os.path.join(output_path, 'cluster_mapping_with_coords.txt')

    with open(cluster_mapping_file, 'w', encoding='utf-8') as f:
        # 写入表头
        f.write("sample_name\tfc1_connectivity\tfc2_pore_size\tfc3_throat_size\t")
        f.write("prop_isolated\tprop_dead_end\tprop_well_connected\tprop_highly_connected\t")
        f.write("prop_nanopore\tprop_micropore\tprop_mesopore\tprop_macropore\t")
        f.write(
            "prop_nano_throat_dominant\tprop_micro_throat_dominant\tprop_meso_throat_dominant\tprop_macro_throat_dominant\n")

        # 遍历每个样本
        for i, sample_name in enumerate(results_df.index):
            # 构建样本名称

            # import pdb;
            # pdb.set_trace()
            row = results_df.loc[sample_name]
            # import pdb;pdb.set_trace()
            # 三元坐标
            fc1 = weighted_features[i,0]
            fc2 = weighted_features[i, 1]
            fc3 = weighted_features[i, 2]

            # 写入数据
            f.write(f"{sample_name}\t{fc1:.6f}\t{fc2:.6f}\t{fc3:.6f}\t")
            f.write(
                f"{row['prop_isolated']:.3f}\t{row['prop_dead_end']:.3f}\t{row['prop_well_connected']:.3f}\t{row['prop_highly_connected']:.3f}\t")
            f.write(
                f"{row['prop_nanopore']:.3f}\t{row['prop_micropore']:.3f}\t{row['prop_mesopore']:.3f}\t{row['prop_macropore']:.3f}\t")
            f.write(
                f"{row['prop_nano_throat_dominant']:.3f}\t{row['prop_micro_throat_dominant']:.3f}\t{row['prop_meso_throat_dominant']:.3f}\t{row['prop_macro_throat_dominant']:.3f}\n")

    logging.info(f"聚类映射文件(含三元坐标)已保存至: {cluster_mapping_file}")

def load_topology_data(topology_file):
    """修复数据加载 - 处理末尾多余的空列"""
    try:
        logging.info("=== 修复数据加载（处理末尾空列）===")

        # 方法1：先读取所有行，清理末尾的\t
        with open(topology_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 清理每行末尾的\t
        cleaned_lines = []
        for line in lines:
            # 去除行尾的\t和空格
            cleaned_line = line.rstrip('\n').rstrip('\t').strip()
            if cleaned_line:  # 只保留非空行
                cleaned_lines.append(cleaned_line)

        # 保存到临时文件
        temp_file = 'temp_cleaned.txt'
        with open(temp_file, 'w', encoding='utf-8') as f:
            for line in cleaned_lines:
                f.write(line + '\n')

        # 读取清理后的文件
        df = pd.read_csv(temp_file, sep='\t')

        # 删除临时文件
        os.remove(temp_file)

        logging.info(f"成功加载数据，共 {len(df)} 个样本")
        logging.info(f"列名: {list(df.columns)}")
        logging.info(f"列数: {len(df.columns)}")

        # 打印前几行验证
        logging.info("前3行数据验证:")
        for i in range(min(3, len(df))):
            logging.info(f"  行{i}: name='{df.iloc[i]['name']}'")

        # 设置索引
        if 'name' in df.columns:
            df = df.set_index('name')
            logging.info(f"已将'name'列设置为索引")
            logging.info(f"前5个索引: {df.index[:5].tolist()}")
        else:
            logging.error("没有找到name列")
            return None, None, None, None

        # 提取特征数据
        coord_features = ['prop_isolated', 'prop_dead_end', 'prop_well_connected', 'prop_highly_connected']
        pore_features = ['prop_nanopore', 'prop_micropore', 'prop_mesopore', 'prop_macropore']
        throat_features = ['prop_nano_throat_dominant', 'prop_micro_throat_dominant',
                           'prop_meso_throat_dominant', 'prop_macro_throat_dominant']

        # 检查特征列是否存在
        for feature_list, name in [(coord_features, '连通性'),
                                   (pore_features, '孔隙尺寸'),
                                   (throat_features, '喉道尺寸')]:
            missing = [col for col in feature_list if col not in df.columns]
            if missing:
                logging.error(f"{name}特征列缺失: {missing}")
                return None, None, None, None

        # 提取数据
        coord_data = df[coord_features].values
        pore_data = df[pore_features].values
        throat_data = df[throat_features].values

        logging.info(f"数据形状: coord={coord_data.shape}, pore={pore_data.shape}, throat={throat_data.shape}")

        return coord_data, pore_data, throat_data, df

    except Exception as e:
        logging.error(f"加载失败: {e}")
        return None, None, None, None
def save_ternary_coordinates(weighted_features, sample_names, output_path, filename="ternary_coordinates.csv"):
    """
    保存三端图的三个维度值（fc1, fc2, fc3）
    """
    if weighted_features is None or len(weighted_features) == 0:
        logging.warning("没有三元坐标数据可保存")
        return

    # 创建DataFrame
    ternary_df = pd.DataFrame({
        'sample_name': sample_names,
        'fc1_connectivity': weighted_features[:, 0],  # 连通性维度
        'fc2_pore_size': weighted_features[:, 1],  # 孔隙尺寸维度
        'fc3_throat_size': weighted_features[:, 2],  # 喉道尺寸维度
        'total_sum': np.sum(weighted_features, axis=1)  # 验证和为1
    })

    # 保存到文件
    output_file = os.path.join(output_path, filename)
    ternary_df.to_csv(output_file, index=False, encoding='utf-8')

    logging.info(f"三元坐标已保存至: {output_file}")
    logging.info(f"共保存了 {len(ternary_df)} 个样本的三元坐标")

    # 打印统计信息
    logging.info("三元坐标统计:")
    logging.info(
        f"  fc1_connectivity: [{ternary_df['fc1_connectivity'].min():.4f}, {ternary_df['fc1_connectivity'].max():.4f}]")
    logging.info(f"  fc2_pore_size: [{ternary_df['fc2_pore_size'].min():.4f}, {ternary_df['fc2_pore_size'].max():.4f}]")
    logging.info(
        f"  fc3_throat_size: [{ternary_df['fc3_throat_size'].min():.4f}, {ternary_df['fc3_throat_size'].max():.4f}]")
    logging.info(f"  所有样本的fc1+fc2+fc3平均值: {ternary_df['total_sum'].mean():.6f}")

    # 同时保存为txt格式
    txt_file = os.path.join(output_path, 'ternary_coordinates.txt')
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("sample_name\tfc1_connectivity\tfc2_pore_size\tfc3_throat_size\n")
        for idx, row in ternary_df.iterrows():
            f.write(
                f"{row['sample_name']}\t{row['fc1_connectivity']:.6f}\t{row['fc2_pore_size']:.6f}\t{row['fc3_throat_size']:.6f}\n")

    logging.info(f"三元坐标(TXT格式)已保存至: {txt_file}")

def perform_rock_clustering_auto(features, method='comprehensive', max_k=15):
    """基于三元坐标进行聚类分析"""
    # features已经是归一化的三元坐标，不需要再标准化
    features_scaled = features  # 三元坐标已经归一化

    n_samples = len(features_scaled)
    max_k = min(max_k, n_samples - 1)
    # 7 最好
    # 您指定了optimal_k=2，所以跳过自动确定
    evaluation_results = 0
    optimal_k = 7

    # 使用K-means进行聚类
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_scaled)

    # 计算聚类质量指标
    silhouette_avg = silhouette_score(features_scaled, cluster_labels)
    calinski_harabasz = calinski_harabasz_score(features_scaled, cluster_labels)
    davies_bouldin = davies_bouldin_score(features_scaled, cluster_labels)

    logging.info(f"\n=== 最终聚类结果 ===")
    logging.info(f"最优聚类数量: {optimal_k}")
    logging.info(f"Silhouette Score: {silhouette_avg:.3f}")
    logging.info(f"Calinski-Harabasz Score: {calinski_harabasz:.3f}")
    logging.info(f"Davies-Bouldin Score: {davies_bouldin:.3f}")

    # 打印每个聚类的样本数
    unique_labels = np.unique(cluster_labels)
    for label in unique_labels:
        count = np.sum(cluster_labels == label)
        percentage = count / len(cluster_labels) * 100
        logging.info(f"类别 {label}: {count} 个岩心 ({percentage:.1f}%)")

    return cluster_labels, kmeans.cluster_centers_, None, optimal_k, evaluation_results
def plot_enhanced_ternary(weighted_features, cluster_labels, pca_info, output_path, sample_names=None):
    """
    增强的三元图可视化 - 添加样本名称标签
    """
    if len(weighted_features) == 0:
        logging.warning("No data points to plot")
        return

    # 使用fc1, fc2, fc3作为维度名称
    fc1 = weighted_features[:, 0]  # 连通性维度
    fc2 = weighted_features[:, 1]  # 孔隙尺寸维度
    fc3 = weighted_features[:, 2]  # 喉道尺寸维度

    # 转换为三角坐标
    x = 0.5 * (2 * fc2 + fc1) / (fc1 + fc2 + fc3 + 1e-8)
    y = (np.sqrt(3) / 2) * fc1 / (fc1 + fc2 + fc3 + 1e-8)

    # 创建图形
    fig = plt.figure(figsize=(18, 14))
    gs = plt.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 1])

    # 主三元图
    ax_main = fig.add_subplot(gs[:, 0])

    # 绘制三角形边界和网格
    triangle = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])
    ax_main.plot(triangle[:, 0], triangle[:, 1], 'k-', linewidth=3)

    # 绘制网格线
    for i in [0.25,0.5,0.75]:
        ax_main.plot([i / 2, 1 - i / 2], [i * np.sqrt(3) / 2, i * np.sqrt(3) / 2],
                     'gray', alpha=0.4, linewidth=1, linestyle='--')
        ax_main.plot([i, 0.5 + i / 2], [0, (1 - i) * np.sqrt(3) / 2],
                     'gray', alpha=0.4, linewidth=1, linestyle='--')
        ax_main.plot([1 - i, 0.5 - i / 2], [0, (1 - i) * np.sqrt(3) / 2],
                     'gray', alpha=0.4, linewidth=1, linestyle='--')

    # 为每个聚类设置颜色和样式
    unique_labels = np.unique(cluster_labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

    # 绘制散点并添加标签
    for i, label in enumerate(unique_labels):
        mask = cluster_labels == label
        if np.sum(mask) > 0:
            # 绘制散点
            scatter = ax_main.scatter(x[mask], y[mask], s=120, color=colors[i],
                                      marker=markers[i % len(markers)],
                                      label=f'Cluster {label} ({np.sum(mask)}个样本)',
                                      alpha=0.7, edgecolors='w', linewidth=1.5, zorder=5)

            # 添加样本名称标签
            if sample_names is not None:
                for j, (x_pos, y_pos) in enumerate(zip(x[mask], y[mask])):
                    idx = np.where(mask)[0][j]
                    sample_name = sample_names[idx]

                    # 根据点的位置调整标签位置，避免重叠
                    offset_x = 0.02
                    offset_y = 0.02

                    # 根据点的位置微调标签位置
                    if x_pos > 0.7:  # 右侧的点
                        offset_x = -0.03
                    if y_pos > 0.7:  # 上方的点
                        offset_y = -0.03

                    # 添加标签
                    ax_main.annotate(sample_name,
                                     xy=(x_pos, y_pos),
                                     xytext=(offset_x, offset_y),
                                     textcoords='offset points',
                                     fontsize=8,
                                     color=colors[i],
                                     alpha=0.8,
                                     bbox=dict(boxstyle='round,pad=0.2',
                                               facecolor='white',
                                               edgecolor=colors[i],
                                               alpha=0.7,
                                               linewidth=0.5),
                                     zorder=10)

    # 设置标签和标题（使用fc1, fc2, fc3）
    ax_main.text(0.5, -0.1, 'fc1: 连通性\n(Connectivity)', ha='center', va='top',
                 fontsize=14, fontweight='bold')
    ax_main.text(0.95, 0.5, 'fc3: 喉道尺寸\n(Throat Size)', ha='left', va='center',
                 fontsize=14, rotation=-60, fontweight='bold')
    ax_main.text(0.05, 0.5, 'fc2: 孔隙尺寸\n(Pore Size)', ha='right', va='center',
                 fontsize=14, rotation=60, fontweight='bold')

    ax_main.set_aspect('equal')
    ax_main.axis('off')
    ax_main.set_title('基于PCA融合的孔隙网络三元图\n带有样本名称标签',
                      fontsize=16, pad=20, fontweight='bold')

    # 添加图例
    ax_main.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center',
                   ncol=min(4, len(unique_labels)), fontsize=10, frameon=True)

    # 可选：添加一个简单版本（无标签的图）
    fig2 = plt.figure(figsize=(12, 10))
    ax2 = fig2.add_subplot(111)
    ax2.plot(triangle[:, 0], triangle[:, 1], 'k-', linewidth=3)

    # 绘制简单散点图
    for i, label in enumerate(unique_labels):
        mask = cluster_labels == label
        if np.sum(mask) > 0:
            ax2.scatter(x[mask], y[mask], s=80, color=colors[i],
                        marker=markers[i % len(markers)],
                        label=f'Cluster {label} ({np.sum(mask)}个样本)',
                        alpha=0.8, edgecolors='w', linewidth=1.5)

    # 设置简单图的标签
    ax2.text(0.5, -0.1, 'fc1: 连通性', ha='center', va='top', fontsize=12)
    ax2.text(0.95, 0.5, 'fc3: 喉道尺寸', ha='left', va='center', fontsize=12, rotation=-60)
    ax2.text(0.05, 0.5, 'fc2: 孔隙尺寸', ha='right', va='center', fontsize=12, rotation=60)

    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title('孔隙网络三元图（无标签版本）', fontsize=14, pad=20)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=min(4, len(unique_labels)))

    # 保存两个版本的图
    filename_with_labels = 'enhanced_ternary_with_labels.png'
    filename_simple = 'ternary_simple.png'

    plt.figure(fig.number)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, filename_with_labels), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_path, 'enhanced_ternary_with_labels.pdf'), bbox_inches='tight')
    plt.show()

    plt.figure(fig2.number)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, filename_simple), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_path, 'ternary_simple.pdf'), bbox_inches='tight')
    plt.show()

    logging.info(f"带标签的三元图已保存至: {os.path.join(output_path, filename_with_labels)}")
    logging.info(f"简单三元图已保存至: {os.path.join(output_path, filename_simple)}")


def plot_ternary_distribution(weighted_features, output_path):
    """
    绘制三元坐标分布图
    """
    if len(weighted_features) == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. 三维散点图
    ax1 = fig.add_subplot(221, projection='3d')
    fc1 = weighted_features[:, 0]
    fc2 = weighted_features[:, 1]
    fc3 = weighted_features[:, 2]

    scatter = ax1.scatter(fc1, fc2, fc3, c=np.arange(len(fc1)), cmap='viridis',
                          s=20, alpha=0.6, edgecolor='k', linewidth=0.5)
    ax1.set_xlabel('fc1: Connectivity', fontsize=10)
    ax1.set_ylabel('fc2: Pore Size', fontsize=10)
    ax1.set_zlabel('fc3: Throat Size', fontsize=10)
    ax1.set_title('Ternary Coordinates 3D Distribution', fontsize=12)

    # 2. fc1 vs fc2
    ax2 = plt.subplot(222)
    ax2.scatter(fc1, fc2, s=20, alpha=0.6, edgecolor='k', linewidth=0.5)
    ax2.set_xlabel('fc1: Connectivity')
    ax2.set_ylabel('fc2: Pore Size')
    ax2.set_title('Connectivity vs Pore Size')
    ax2.grid(True, alpha=0.3)

    # 3. fc1 vs fc3
    ax3 = plt.subplot(223)
    ax3.scatter(fc1, fc3, s=20, alpha=0.6, edgecolor='k', linewidth=0.5)
    ax3.set_xlabel('fc1: Connectivity')
    ax3.set_ylabel('fc3: Throat Size')
    ax3.set_title('Connectivity vs Throat Size')
    ax3.grid(True, alpha=0.3)

    # 4. fc2 vs fc3
    ax4 = plt.subplot(224)
    ax4.scatter(fc2, fc3, s=20, alpha=0.6, edgecolor='k', linewidth=0.5)
    ax4.set_xlabel('fc2: Pore Size')
    ax4.set_ylabel('fc3: Throat Size')
    ax4.set_title('Pore Size vs Throat Size')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'ternary_coordinates_distribution.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_path, 'ternary_coordinates_distribution.pdf'), bbox_inches='tight')
    plt.show()

    logging.info(f"三元坐标分布图已保存至: {output_path}")
def save_cluster_samples_list(results_df, output_path):
    """
    保存每个聚类类别包含的样本列表 - 详细格式
    """
    cluster_samples_file = os.path.join(output_path, 'cluster_samples_list.txt')

    with open(cluster_samples_file, 'w', encoding='utf-8') as f:
        # 按聚类类别分组
        clusters = results_df.groupby('cluster_id')

        f.write("=== 聚类样本详细列表 ===\n\n")

        for cluster_id, group in clusters:
            f.write(f"=== 类别 {cluster_id} ===\n")
            f.write(f"样本数量: {len(group)} 个\n")
            f.write(f"占比: {len(group) / len(results_df) * 100:.1f}%\n\n")

            # 获取样本名称
            sample_names = group.index.tolist()
            sample_names.sort()

            f.write("样本名称列表:\n")
            for i, sample_name in enumerate(sample_names):
                f.write(f"  {i + 1:3d}. {sample_name}\n")

            f.write("\n")

            # 显示每个样本的三元坐标
            f.write("样本的三元坐标:\n")
            for sample_name in sample_names[:10]:  # 只显示前10个
                row = results_df.loc[sample_name]
                f.write(f"  {sample_name}: ")
                f.write(f"fc1={row['fc1_connectivity']:.3f}, ")
                f.write(f"fc2={row['fc2_pore_size']:.3f}, ")
                f.write(f"fc3={row['fc3_throat_size']:.3f}\n")

            if len(sample_names) > 10:
                f.write(f"  ... 还有 {len(sample_names) - 10} 个样本\n")

            f.write("\n" + "=" * 50 + "\n\n")

    logging.info(f"聚类样本列表文件已保存至: {cluster_samples_file}")
def main_analysis_with_model_saving():
    """主分析函数 - 增加模型保存功能"""
    res = 128
    topology_file = f'./sub_txt/topology.txt'
    output_path = f'./pca_result/best_model_saving_{res}'
    os.makedirs(output_path, exist_ok=True)

    # 1. 加载数据
    logging.info("=== 加载拓扑数据 ===")
    coord_data, pore_data, throat_data, original_df = load_topology_data(topology_file)
    if coord_data is None:
        return

    # 2. 提取样本名称
    sample_names = original_df.index.tolist()

    # 3. 创建特征并进行PCA融合，同时保存模型
    logging.info("=== 进行PCA维度融合并保存模型 ===")
    composite_features, weighted_features, pca_info = create_composite_features_pca(
        coord_data, pore_data, throat_data,
        save_model=True,  # 开启模型保存
        output_path=output_path
    )

    # 4. 保存三元坐标
    logging.info("=== 保存三元坐标 ===")
    save_ternary_coordinates(weighted_features, sample_names, output_path)

    # 5. 绘制三元坐标分布图
    logging.info("=== 绘制三元坐标分布图 ===")
    plot_ternary_distribution(weighted_features, output_path)

    # 6. 基于三元坐标进行聚类
    if len(composite_features) > 1:
        logging.info("=== 基于三元坐标进行聚类分析 ===")
        cluster_labels, cluster_centers, scaler, optimal_k, eval_results = perform_rock_clustering_auto(
            composite_features, method='comprehensive', max_k=min(15, len(composite_features) // 2)
        )

        # 7. 保存聚类结果
        results_df = original_df.copy()
        results_df['cluster_id'] = cluster_labels
        results_df['fc1_connectivity'] = weighted_features[:, 0]
        results_df['fc2_pore_size'] = weighted_features[:, 1]
        results_df['fc3_throat_size'] = weighted_features[:, 2]

        # 保存CSV结果
        results_df.to_csv(os.path.join(output_path, 'rock_classification_results.csv'), index=True)

        # 8. 保存聚类映射
        save_cluster_mapping_with_coordinates(results_df, weighted_features, output_path)

        # 9. 保存详细的聚类样本列表
        logging.info("=== 保存聚类样本列表 ===")
        save_cluster_samples_list(results_df, output_path)

        # 10. 生成增强的可视化
        plot_enhanced_ternary(weighted_features, cluster_labels, pca_info, output_path, sample_names=sample_names)

        # 11. 保存聚类中心信息
        cluster_centers_file = os.path.join(output_path, 'cluster_centers.json')
        cluster_info = {
            'optimal_k': int(optimal_k),
            'cluster_centers': cluster_centers.tolist(),
            'cluster_sizes': [int(np.sum(cluster_labels == i)) for i in range(optimal_k)],
            'cluster_percentages': [float(np.sum(cluster_labels == i) / len(cluster_labels) * 100)
                                    for i in range(optimal_k)]
        }

        with open(cluster_centers_file, 'w', encoding='utf-8') as f:
            json.dump(cluster_info, f, indent=2, ensure_ascii=False)

        logging.info(f"聚类中心信息已保存至: {cluster_centers_file}")

    # 12. 创建模型使用示例
    create_model_usage_example(output_path)

    logging.info(f"所有分析完成！结果保存在: {output_path}")


def create_model_usage_example(output_path):
    """创建模型使用示例"""
    example_file = os.path.join(output_path, 'model_usage_example.py')

    example_code = '''"""
孔隙拓扑模型使用示例
使用方法:
1. 加载保存的模型（包含固化的 StandardScaler, PCA 权重, 和 PxTyCz 分类边界）
2. 对新数据进行分类，直接获取三元坐标和拓扑标签。
"""

import numpy as np
import pandas as pd
import os
# 确保你的 python 环境中能够导入 PoreTopologyModel
# 假设你的 PoreTopologyModel 类位于当前路径或已在 PYTHONPATH 中
# from . import PoreTopologyModel # 如果是包内导入
# PoreTopologyModel is defined in this module

def load_and_apply_model():
    # 1. 加载模型
    # 假设模型文件位于 output_path 下
    model_path = os.path.join(os.path.dirname(__file__), 'pore_topology_model.pkl') 

    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误：模型文件未找到于 {model_path}。请先运行主分析函数保存模型。")
        return

    model = PoreTopologyModel()
    model.load_model(model_path)  # 加载固化的模型参数

    # 2. 准备新数据（示例）
    # 新数据应该包含训练模型时的所有12个特征列
    new_data = pd.DataFrame({
        'prop_isolated': [8.936, 7.415, 6.545],
        'prop_dead_end': [39.149, 42.797, 39.455],
        'prop_well_connected': [34.894, 37.712, 40.909],
        'prop_highly_connected': [17.021, 12.076, 13.091],
        'prop_nanopore': [13.404, 12.076, 12.182],
        'prop_micropore': [64.681, 67.373, 69.455],
        'prop_mesopore': [19.787, 19.068, 18.000],
        'prop_macropore': [2.128, 1.483, 0.364],
        'prop_nano_throat_dominant': [0.000, 0.000, 0.000],
        'prop_micro_throat_dominant': [0.000, 0.000, 0.000],
        'prop_meso_throat_dominant': [14.252, 13.501, 14.008],
        'prop_macro_throat_dominant': [85.748, 86.499, 85.992]
    })

    # 为了演示，给新样本加上名称
    new_data.index = ['Sample_New_A', 'Sample_New_B', 'Sample_New_C']

    # 3. 核心步骤：应用模型获取三元坐标和拓扑类型
    # model.apply_to_new_data 现在返回 (final_coords, categories)
    ternary_coords, categories = model.apply_to_new_data(new_data)

    print("\\n=== 投影结果 ===")
    print("三元坐标 (fc1=连通性, fc2=孔隙尺寸, fc3=喉道尺寸):")

    results_df = pd.DataFrame(ternary_coords, columns=['fc1', 'fc2', 'fc3'], index=new_data.index)
    results_df['Topology_Category'] = categories

    print(results_df.round(4))

    print("\\n拓扑类型 (PxTyCz):")
    print(results_df['Topology_Category'].tolist())

    # 可选：显示分类边界（证明使用了固化参数）
    if model.normalization_params:
        print("\\n使用的固化分类边界 (25%, 50%, 75% 分位数):")
        print(model.normalization_params['classification_boundaries'])


if __name__ == "__main__":
    # 为了在实际运行时能够找到模型，我们需要一个假的 PoreTopologyModel 类
    # 但在实际应用中，用户需要将真正的 PoreTopologyModel 类与示例放在一起
    try:
        load_and_apply_model()
    except Exception as e:
        print(f"运行示例失败: {e}")
        print("请确保 pore_topology_model.py 和 pore_topology_model.pkl/json 在同一目录下。")
'''

    with open(example_file, 'w', encoding='utf-8') as f:
        f.write(example_code)

    logging.info(f"模型使用示例已保存至: {example_file}")
# 其他函数保持不变...

if __name__ == "__main__":
    # 运行增强版本（带模型保存）
    main_analysis_with_model_saving()