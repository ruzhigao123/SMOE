"""
训练数据生成脚本

本脚本的主要功能：
1. 从 new_kernel/dse_database（gexf + .db + sources / config）读取数据
2. 对每个kernel的每个配置点进行特征编码（节点特征、边特征）
3. 构建PyTorch Geometric的Data对象，包含图结构、特征和目标值
4. 保存处理后的数据到 two_tower_datasets_c 目录（不存在则创建）
5. 对于配置，生成并保存替换后的 .c 文件到 code/ 目录（不再使用 CodeBERT 编码）

数据流程：
- 输入：new_kernel/dse_database/{machsuite,poly,programl}
- 处理：特征编码 + 生成 .c 文件
- 输出：two_tower_datasets_c/graph/ 和 two_tower_datasets_c/code/ 目录下的文件（.pt 和 .c）
"""

# ---- multimodal_moe path bootstrap (HMoE dataset) ----
import os, sys
from os.path import abspath, dirname, join
_DSE_GEN_DIR = dirname(abspath(__file__))
_MM_ROOT = dirname(_DSE_GEN_DIR)                 # .../multimodal_moe
_HMOE_ROOT = dirname(_MM_ROOT)                   # .../HierarchicalMoE-master
_HMOE_SRC = join(_HMOE_ROOT, 'src')
# HMoE 的 config.py 使用「from utils import ...」（需 src 在 path）
# 同时「import src.config」需要仓库根在 path
for _p in (_HMOE_SRC, _HMOE_ROOT, _DSE_GEN_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# -----------------------------------------------------

import os.path as osp
from os.path import join, basename
import sys
import os

# 添加项目根目录到Python路径，以便导入src模块
import src.config as config
from src.config import FLAGS
from glob import glob, iglob
from src.utils import get_root_path, MLP, print_stats, get_save_path, \
    create_dir_if_not_exists, plot_dist, load
from collections import Counter, OrderedDict
import pickle
from sklearn.preprocessing import OneHotEncoder
import warnings
from tqdm import tqdm
from src.saver import saver
import networkx as nx
from programl_data import _encode_X_dict, _encode_edge_dict, _encode_X_torch, _encode_edge_torch, \
    finte_diff_as_quality, create_edge_index
from hlsyn_io import (
    DSE_DB_ROOT, GRAPHS_DIR,
    resolve_gexf, resolve_source, load_kernel_database, apply_point_to_source,
)
import math
import torch
import numpy as np
from shutil import rmtree
from torch_geometric.data import Data
from copy import deepcopy
from torch_geometric.data import Dataset
from pathlib import Path

# 数据标签标识
tag = 'new_speedup'

# 从配置文件读取目标变量和kernel列表
TARGETS = config.TARGETS  # 训练目标，如perf, quality, util-BRAM等
MACHSUITE_KERNEL = config.MACHSUITE_KERNEL  # MachSuite基准测试的kernel列表
poly_KERNEL = config.poly_KERNEL  # PolyBench基准测试的kernel列表

# 使用 HMoE / HLSyn 全量 kernel（覆盖 config 中可能为空的 MACHSUITE）
MACHSUITE_KERNEL = ['aes', 'gemm-blocked', 'gemm-ncubed', 'spmv-crs', 'spmv-ellpack', 'stencil', 'nw', 'md', 'stencil-3d']
poly_KERNEL = ['2mm', '3mm', 'adi', 'atax', 'bicg', 'bicg-large', 'covariance', 'doitgen',
               'doitgen-red', 'fdtd-2d', 'fdtd-2d-large', 'gemm-p', 'gemm-p-large', 'gemver',
               'gesummv', 'heat-3d', 'jacobi-1d', 'jacobi-2d', 'mvt', 'seidel-2d', 'symm',
               'symm-opt', 'syrk', 'syr2k', 'trmm', 'trmm-opt', 'mvt-medium', 'correlation',
               'atax-medium', 'bicg-medium', 'gesummv-medium', 'symm-opt-medium', 'gemver-medium']
ALL_KERNEL = MACHSUITE_KERNEL + poly_KERNEL

# ==================== 路径配置（new_kernel/dse_database）====================
# 与 gen_dataset.py 的 two_tower_datasets 分开，避免互相覆盖
_DATASET_ROOT = join(_MM_ROOT, 'two_tower_datasets_c')
# 代码保存路径（替换后的 .c 文件）
code_path = join(_DATASET_ROOT, 'code')
# 图数据保存路径（PyTorch Geometric Data对象）
graph_path = join(_DATASET_ROOT, 'graph')
# points_list.pkl 保存路径（所有 kernel 的 design_point 列表）
points_path = join(_DATASET_ROOT, 'points')
# 编码器保存路径（OneHotEncoder等）
ENCODER_PATH = join(_HMOE_ROOT, 'encoders')
# 根目录不存在则创建
create_dir_if_not_exists(_DATASET_ROOT)
create_dir_if_not_exists(code_path)
create_dir_if_not_exists(graph_path)
create_dir_if_not_exists(points_path)

# new_kernel：按 ALL_KERNEL 解析 gexf + .db
GEXF_FOLDER = GRAPHS_DIR
GEXF_FILES = []
for _k in ALL_KERNEL:
    _gf = resolve_gexf(_k)
    if _gf:
        GEXF_FILES.append((_k, _gf))
_graph_type = "new_kernel-dse_database"

if __name__ == '__main__':
    # ==================== 初始化 ====================
    # 使用Python字典替代Redis（用于临时存储和查询.db文件中的数据）
    # 注意：原代码使用Redis，但为了简化部署，改用内存字典
    database = {}  # 使用字典存储数据，key为配置点名称，value为pickle序列化的对象
    print(graph_path)
    bench = ['machsuite', 'poly']
    saver.log_info(
        f'new_kernel data root={DSE_DB_ROOT}; '
        f'{len(GEXF_FILES)}/{len(ALL_KERNEL)} kernels with gexf (graph_type={_graph_type!r})'
    )

    # ==================== 特征统计计数器 ====================
    # 用于统计所有出现的特征类型，以便后续进行OneHot编码
    ntypes = Counter()  # 节点类型统计
    ptypes = Counter()  # pragma类型统计
    numerics = Counter()  # 数值特征统计
    itypes = Counter()  # 整数类型统计
    ftypes = Counter()  # 浮点类型统计
    btypes = Counter()  # 布尔类型统计
    ptypes_edge = Counter()  # 边的pragma类型统计
    ftypes_edge = Counter()  # 边的浮点类型统计

    # ==================== 加载或创建编码器 ====================
    # 如果提供了预训练的编码器路径，则加载；否则创建新的OneHot编码器
    encoders_loaded = False
    if FLAGS.encoder_path != None:
        try:
            # 加载已保存的编码器（用于增量训练或使用已有编码）
            # 注意：不同 sklearn 版本之间反序列化 OneHotEncoder 可能会产生 InconsistentVersionWarning。
            # 一般只影响"复用旧 encoder"场景；如果你想彻底消除风险/警告，直接把 FLAGS.encoder_path 设为 None 重新生成即可。
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                encoders = load(FLAGS.encoder_path)


            # 修复 sklearn 版本兼容性问题：旧版本保存的编码器可能缺少新版本的属性
            def _fix_encoder_compat(encoder):
                """修复旧版本 OneHotEncoder 的兼容性问题"""
                # sklearn 1.0+ 新增的私有属性，旧版本没有，可以安全设置
                if not hasattr(encoder, '_infrequent_enabled'):
                    try:
                        encoder._infrequent_enabled = False
                    except:
                        pass
                # 对于只读属性，使用 object.__setattr__ 绕过属性检查
                if not hasattr(encoder, '_infrequent_indices'):
                    try:
                        object.__setattr__(encoder, '_infrequent_indices', None)
                    except:
                        pass  # 如果设置失败，忽略
                # infrequent_categories_ 是只读属性，尝试通过 __dict__ 设置（如果失败就忽略）
                if not hasattr(encoder, 'infrequent_categories_'):
                    try:
                        # 使用 object.__setattr__ 绕过只读属性检查
                        object.__setattr__(encoder, 'infrequent_categories_', [])
                    except:
                        # 如果还是失败，尝试通过 __dict__ 直接设置
                        try:
                            encoder.__dict__['infrequent_categories_'] = []
                        except:
                            pass  # 如果都失败，忽略（可能在使用时会有问题，但至少不会立即崩溃）
                return encoder


            enc_ntype = _fix_encoder_compat(encoders['enc_ntype'])
            enc_ptype = _fix_encoder_compat(encoders['enc_ptype'])
            enc_itype = _fix_encoder_compat(encoders['enc_itype'])
            enc_ftype = _fix_encoder_compat(encoders['enc_ftype'])
            enc_btype = _fix_encoder_compat(encoders['enc_btype'])

            enc_ftype_edge = _fix_encoder_compat(encoders['enc_ftype_edge'])
            enc_ptype_edge = _fix_encoder_compat(encoders['enc_ptype_edge'])
            encoders_loaded = True
            saver.log_info(f"Successfully loaded encoders from {FLAGS.encoder_path}")
        except Exception as e:
            # 如果加载或修复编码器失败，回退到创建新编码器
            saver.warning(f"Failed to load/fix encoders from {FLAGS.encoder_path}: {e}")
            saver.warning("Falling back to creating new encoders...")
            encoders_loaded = False

    if not encoders_loaded:
        # 创建新的OneHot编码器
        # handle_unknown='ignore' 对于处理新kernel中未知的变量类型至关重要
        enc_ntype = OneHotEncoder(handle_unknown='ignore')
        enc_ptype = OneHotEncoder(handle_unknown='ignore')
        enc_itype = OneHotEncoder(handle_unknown='ignore')
        enc_ftype = OneHotEncoder(handle_unknown='ignore')
        enc_btype = OneHotEncoder(handle_unknown='ignore')

        enc_ftype_edge = OneHotEncoder(handle_unknown='ignore')
        enc_ptype_edge = OneHotEncoder(handle_unknown='ignore')
        saver.log_info("Created new OneHot encoders")

    # ==================== 数据存储变量 ====================
    data_list = []  # 存储处理后的Data对象列表
    all_gs = OrderedDict()  # 存储所有图对象，key为kernel名称

    # 收集所有特征用于编码器拟合
    X_ntype_all = []  # 所有节点的类型特征
    X_ptype_all = []  # 所有节点的pragma类型特征
    X_itype_all = []  # 所有节点的整数类型特征
    X_ftype_all = []  # 所有节点的浮点类型特征
    X_btype_all = []  # 所有节点的布尔类型特征

    edge_ftype_all = []  # 所有边的浮点类型特征
    edge_ptype_all = []  # 所有边的pragma类型特征

    tot_configs = 0  # 总配置数量统计
    num_files = 0  # 处理的文件数量统计
    init_feat_dict = {}  # 初始特征字典

    # ==================== 主循环：按 ALL_KERNEL 处理 new_kernel/.db ====================
    skipped_kernels = []  # 记录被跳过的kernel
    for matched_kernel, gexf_file in tqdm(GEXF_FILES):
        n = matched_kernel

        try:
            g = nx.read_gexf(gexf_file)
        except Exception as e:
            saver.warning(f'[SKIP] 无法读取 gexf: {gexf_file} ({type(e).__name__}: {e})')
            skipped_kernels.append(n)
            continue
        g.variants = OrderedDict()
        gname = basename(gexf_file).split('.')[0]
        saver.log_info(f'{n} <- {gname}')
        all_gs[n] = g

        database.clear()
        try:
            loaded, design_path, design_ver = load_kernel_database(n)
            database.update(loaded)
            saver.log_info(
                f'.db designs for {n}: {design_path} ({design_ver}), '
                f'{len(database)} points'
            )
        except Exception as e:
            saver.warning(f'[SKIP] No .db designs for {n}: {e}')
            skipped_kernels.append(n)
            continue

        keys = list(database.keys())
        lv2_keys = [k for k in keys if 'lv2' in k]
        saver.log_info(f'num keys for {n}: {len(keys)} and lv2 keys: {len(lv2_keys)}')

        got_reference = False
        res_reference = None
        max_perf = 0.0
        min_lat = float(getattr(FLAGS, "min_allowed_latency", 1.0) or 1.0)
        for key in sorted(keys):
            obj = database[key]
            if key[0:3] == 'lv1' or float(getattr(obj, "perf", 0) or 0) <= 0:
                continue
            if float(obj.perf) > max_perf:
                max_perf = float(obj.perf)
                got_reference = True
                res_reference = obj
        if got_reference and res_reference is not None:
            saver.log_info(f'reference point for {n} is {res_reference.perf}')
        else:
            saver.warning(
                f'[SKIP] no valid reference (perf>0) for {n} '
                f'with {len(keys)} keys / {len(lv2_keys)} lv2 — skip kernel'
            )
            skipped_kernels.append(n)
            continue

        for key in sorted(keys):
            obj = database[key]
            if FLAGS.task == 'regression' and key[0:3] == 'lv1':
                continue
            if FLAGS.task == 'regression' and (not getattr(FLAGS, "invalid", False)) and float(getattr(obj, "perf", 0) or 0) < min_lat:
                continue
            if getattr(obj, "point", None) == {}:
                continue

            print(obj.point)

            try:
                xy_dict = _encode_X_dict(
                    g, ntypes=ntypes, ptypes=ptypes, itypes=itypes, ftypes=ftypes, btypes=btypes, numerics=numerics,
                    obj=obj)
                edge_dict = _encode_edge_dict(
                    g, ftypes=ftypes_edge, ptypes=ptypes_edge)
            except KeyError as e:
                saver.warning(
                    f'[SKIP] {n} key={key}: design point 与 gexf AUTO pragma 不一致 ({e})'
                )
                continue

            if FLAGS.task == 'regression':
                for tname in TARGETS:
                    if tname == 'perf':
                        if obj.perf > 0:
                            y = FLAGS.normalizer / obj.perf
                            y = math.log(y + 1, 100)
                        else:
                            y = obj.perf
                        xy_dict['actual_perf'] = torch.FloatTensor(np.array([obj.perf]))
                    elif tname == 'quality':
                        y = finte_diff_as_quality(obj, res_reference)
                    elif 'util' in tname or 'total' in tname:
                        y = math.log(obj.res_util[tname] + 1, 100)
                    else:
                        raise NotImplementedError()
                    xy_dict[tname] = torch.FloatTensor(np.array([y]))
            else:
                raise NotImplementedError()

            vname = key
            g.variants[vname] = (xy_dict, edge_dict, obj.point)

            X_ntype_all += xy_dict['X_ntype']
            X_ptype_all += xy_dict['X_ptype']
            X_itype_all += xy_dict['X_itype']
            X_ftype_all += xy_dict['X_ftype']
            X_btype_all += xy_dict['X_btype']

            edge_ftype_all += edge_dict['X_ftype']
            edge_ptype_all += edge_dict['X_ptype']

        tot_configs += len(g.variants)
        num_files += 1
        saver.log_info(f'{n} g.variants {len(g.variants)} tot_configs {tot_configs}')
        saver.log_info(f'\tntypes {len(ntypes)}')
        saver.log_info(f'\tptypes {len(ptypes)} {ptypes}')
        saver.log_info(f'\tnumerics {len(numerics)} {numerics}')

    # ==================== 处理完成统计 ====================
    saver.log_info(f'\n{"=" * 60}')
    saver.log_info(f'[SUMMARY] Processing Summary:')
    saver.log_info(f'  Total .gexf files found: {len(GEXF_FILES)}')
    saver.log_info(f'  Kernels successfully processed: {num_files}')
    saver.log_info(f'  Kernels skipped (not in ALL_KERNEL): {len(skipped_kernels)}')
    if skipped_kernels:
        saver.log_info(f'  Skipped kernels:')
        for sf in skipped_kernels:
            saver.log_info(f'    - {sf}')
    saver.log_info(f'  Expected kernels (MACHSUITE): {len(MACHSUITE_KERNEL)} - {MACHSUITE_KERNEL}')
    saver.log_info(f'  Expected kernels (POLY): {len(poly_KERNEL)} - {poly_KERNEL}')
    saver.log_info(f'  Total expected: {len(ALL_KERNEL)}')
    saver.log_info(f'{"=" * 60}\n')

    # ==================== 拟合编码器（在所有数据收集完成后） ====================
    # 如果创建了新编码器，需要先拟合才能使用
    if not encoders_loaded:
        saver.log_info("Fitting OneHot encoders with collected features...")
        # 将列表转换为numpy数组用于fit（需要是2D数组）
        if len(X_ntype_all) > 0:
            enc_ntype.fit(np.array(X_ntype_all).reshape(-1, 1))
        if len(X_ptype_all) > 0:
            enc_ptype.fit(np.array(X_ptype_all).reshape(-1, 1))
        if len(X_itype_all) > 0:
            enc_itype.fit(np.array(X_itype_all).reshape(-1, 1))
        if len(X_ftype_all) > 0:
            enc_ftype.fit(np.array(X_ftype_all).reshape(-1, 1))
        if len(X_btype_all) > 0:
            enc_btype.fit(np.array(X_btype_all).reshape(-1, 1))
        if len(edge_ftype_all) > 0:
            enc_ftype_edge.fit(np.array(edge_ftype_all).reshape(-1, 1))
        if len(edge_ptype_all) > 0:
            enc_ptype_edge.fit(np.array(edge_ptype_all).reshape(-1, 1))
        saver.log_info("OneHot encoders fitted successfully")

    # ==================== 为每个kernel生成最终数据 ====================
    # 遍历所有已处理的图，生成PyTorch Geometric的Data对象并保存
    if not all_gs:
        saver.error("No graphs found in all_gs! Cannot generate dataset.")
        raise ValueError("all_gs is empty, cannot proceed with dataset generation")

    saver.log_info(f"Generating dataset for {len(all_gs)} kernels")
    for gname, g in all_gs.items():
        # 创建边索引（PyTorch Geometric格式）
        edge_index = create_edge_index(g)
        saver.log_info('edge_index created', gname)
        data_list = []  # 存储当前kernel的所有Data对象

        # ==================== 确定保存路径 ====================
        # 根据kernel所属的benchmark设置不同的保存路径
        if gname in MACHSUITE_KERNEL:
            suite = 'machsuite'
        else:
            suite = 'poly'
        SAVE_DIR = join(graph_path, suite, gname)
        SAVE_DIR_1 = join(code_path, suite, gname)
        CODE_FILES = resolve_source(gname)

        # ==================== 读取源代码文件 ====================
        code_list = []
        if not CODE_FILES or not osp.exists(CODE_FILES):
            saver.warning(
                f'[SKIP] Source code not found for {gname} under new_kernel sources/'
            )
            continue
        try:
            with open(CODE_FILES, 'r', encoding='utf-8') as file:
                fc = file.read()
            saver.log_info(f'Using source file for {gname}: {CODE_FILES}')
        except Exception as e:
            saver.error(f"Failed to read source code file {CODE_FILES}: {e}")
            raise

        # ==================== 创建代码保存目录 ====================
        create_dir_if_not_exists(SAVE_DIR_1)

        # ==================== 处理每个配置变体 ====================
        # 检查是否有变体数据
        if not g.variants:
            saver.warning(f"No variants found for kernel {gname}, skipping...")
            continue

        # 收集所有设计点，用于保存 points_list.pkl
        points_list = []
        points_collected = 0
        points_none_count = 0
        for nums, (vname, d) in enumerate(g.variants.items()):
            d_node, d_edge, point = d  # 解包节点特征、边特征和设计点
            # 收集设计点
            points_list.append(point)
            points_collected += 1
            if point is None:
                points_none_count += 1
            elif nums < 3:  # 只打印前3个样本的调试信息
                saver.log_info(
                    f'[Debug] Collected point for {gname} variant {vname}: {type(point)}, keys: {list(point.keys()) if isinstance(point, dict) else "N/A"}')

        saver.log_info(
            f'[Debug] Collected {points_collected} points for kernel {gname} (None: {points_none_count}, non-None: {points_collected - points_none_count})')

        # 重新遍历 variants 来构建 Data 对象（因为 points_list 已经在上面收集完成）
        for nums, (vname, d) in enumerate(g.variants.items()):
            d_node, d_edge, point = d  # 解包节点特征、边特征和设计点

            # ==================== 转换为PyTorch张量 ====================
            # 使用编码器将特征字典转换为张量格式
            X = _encode_X_torch(d_node, enc_ntype, enc_ptype, enc_itype, enc_ftype, enc_btype)
            edge_attr = _encode_edge_torch(d_edge, enc_ftype_edge, enc_ptype_edge)

            # ==================== 构建PyTorch Geometric Data对象 ====================
            # 包含图结构、节点特征、边特征和目标值
            if FLAGS.task == 'regression':
                data_list.append(Data(
                    x=X,  # 节点特征矩阵
                    edge_index=edge_index,  # 边索引（COO格式）
                    perf=d_node['perf'],  # 归一化的性能目标
                    actual_perf=d_node['actual_perf'],  # 实际性能值
                    quality=d_node['quality'],  # 质量目标
                    util_BRAM=d_node['util-BRAM'],  # BRAM利用率
                    util_DSP=d_node['util-DSP'],  # DSP利用率
                    util_LUT=d_node['util-LUT'],  # LUT利用率
                    util_FF=d_node['util-FF'],  # FF利用率
                    total_BRAM=d_node['total-BRAM'],  # BRAM总量
                    total_DSP=d_node['total-DSP'],  # DSP总量
                    total_LUT=d_node['total-LUT'],  # LUT总量
                    total_FF=d_node['total-FF'],  # FF总量
                    edge_attr=edge_attr,  # 边特征
                    kernel=gname  # kernel名称
                ))
            else:
                raise NotImplementedError()

            # ==================== 生成 .c 文件 ====================
            # 用 design point（.db）替换 auto{PRAGMA}；兼容 lv1:/lv2: key
            fcc = apply_point_to_source(fc, point, key=vname)

            # 保存替换后的 .c 文件
            try:
                pa = join(SAVE_DIR_1, f'{nums}.c')
                with open(pa, 'w', encoding='utf-8') as f:
                    f.write(fcc)
                saver.log_info(f"Saved .c file: {pa}")
            except Exception as e:
                saver.error(f"Failed to save .c file for {gname} variant {vname}: {e}")

        # ==================== 保存数据到磁盘 ====================
        # 使用 try-finally 确保即使出错也保存 points_list.pkl
        try:
            saver.log_info(f'Saving {len(data_list)} to disk {SAVE_DIR}; Deleting existing files')
            # 安全删除目录（如果存在）
            if osp.exists(SAVE_DIR):
                rmtree(SAVE_DIR)
            create_dir_if_not_exists(SAVE_DIR)  # 创建目录
            # 逐个保存Data对象
            for i in tqdm(range(len(data_list))):
                torch.save(data_list[i], osp.join(SAVE_DIR, 'data_{}.pt'.format(i)))
        finally:
            # ==================== 保存 points_list.pkl ====================
            # 保存设计点列表，用于因果模型训练
            # 使用 finally 确保即使保存 data_*.pt 失败，也保存 points_list.pkl
            # 保存到专门的 points 目录：two_tower_dataset/points/{benchmark}/{kernel}/points_list.pkl
            try:
                # 确定 benchmark
                if gname in MACHSUITE_KERNEL:
                    benchmark = 'machsuite'
                else:
                    benchmark = 'poly'

                # 构建保存路径
                kernel_points_dir = join(points_path, benchmark, gname)
                points_list_path = join(kernel_points_dir, 'points_list.pkl')

                # 确保目录存在
                create_dir_if_not_exists(kernel_points_dir)

                # 验证 points_list 和 data_list 的长度是否一致
                if len(points_list) != len(data_list):
                    saver.warning(
                        f'⚠️  points_list length ({len(points_list)}) != data_list length ({len(data_list)}) for kernel {gname}')
                    saver.warning(f'⚠️  This may cause index mismatch. Truncating points_list to match data_list.')
                    # 截断 points_list 以匹配 data_list 的长度
                    points_list = points_list[:len(data_list)]

                # 验证 points_list 不为空
                if len(points_list) == 0:
                    saver.warning(f'points_list is empty for kernel {gname}, skipping points_list.pkl save')
                else:
                    # 保存 points_list
                    with open(points_list_path, 'wb') as f:
                        pickle.dump(points_list, f)

                    # 验证文件是否成功保存
                    if osp.exists(points_list_path):
                        file_size = osp.getsize(points_list_path)
                        non_none_count = sum(1 for p in points_list if p is not None)
                        saver.log_info(
                            f'✓ Saved {len(points_list)} design points to {points_list_path} (non-None: {non_none_count}, file size: {file_size} bytes)')
                        saver.log_info(f'✓ points_list.pkl location: {osp.abspath(points_list_path)}')
                    else:
                        saver.error(f'✗ Failed to save points_list.pkl: file does not exist after save operation')
                        saver.error(f'✗ Expected path: {osp.abspath(points_list_path)}')
            except Exception as e:
                saver.error(f'✗ Failed to save points_list.pkl for kernel {gname}: {e}')
                import traceback

                saver.error(traceback.format_exc())
                # 不抛出异常，继续处理其他 kernel

        # ==================== 统计信息输出 ====================
        # 统计节点数量
        nns = [d.x.shape[0] for d in data_list]
        print_stats(nns, 'number of nodes')
        # 统计平均度数
        ads = [d.edge_index.shape[1] / d.x.shape[0] for d in data_list]
        print_stats(ads, 'avg degrees')
        saver.log_info('dataset[0].num_features', data_list[0].num_features)

        # ==================== 目标值分布分析 ====================
        # 对每个目标变量绘制分布图并统计
        for target in TARGETS:
            if not hasattr(data_list[0], target.replace('-', '_')):
                saver.warning(f'Data does not have attribute {target}')
                continue
            # 提取所有样本的目标值
            ys = [getattr(d, target.replace('-', '_')).item() for d in data_list]
            # if target == 'quality':
            #     continue
            # 绘制分布图
            plot_dist(ys, f'{target}_ys', saver.get_log_dir(), saver=saver, analyze_dist=True, bins=None)
            saver.log_info(f'{target}_ys', Counter(ys))


# ==================== 数据集类（用于训练时加载数据） ====================
class MyOwnDataset():
    """
    自定义数据集类，用于在训练时加载已生成的数据文件
    这个类提供了访问two_tower_dataset目录下数据的方法
    """

    def __init__(self):
        pass

    @property
    def raw_file_names(self):
        """
        返回原始文件列表（未使用）
        """
        # return ['some_file_1', 'some_file_2', ...]
        return []

    @property
    def processed_file_names(self):
        """
        返回所有已处理的数据文件路径
        返回两个字典：
        - gp: 图数据文件路径字典，key为kernel名称，value为文件路径列表
        - cp: 代码文件路径字典，key为kernel名称，value为文件路径列表（.c 文件）
        """
        gp = {}  # graph paths
        cp = {}  # code paths
        for k in ALL_KERNEL:
            if k in MACHSUITE_KERNEL:
                gt = join(graph_path, 'machsuite', k)
                ct = join(code_path, 'machsuite', k)
            else:
                gt = join(graph_path, 'poly', k)
                ct = join(code_path, 'poly', k)
            gp[k] = glob(join(gt, '*.pt'))  # 获取所有.pt图数据文件
            cp[k] = glob(join(ct, '*.c'))  # 获取所有.c 代码文件
        return gp, cp

    def download(self):
        """
        下载数据（未实现，因为数据是本地生成的）
        """
        pass

    # Download to `self.raw_dir`.

    def process(self):
        """
        处理原始数据（未实现，因为数据已经在gen_dataset.py中处理完成）
        """
        # i = 0
        # for raw_path in self.raw_paths:
        #     # Read data from `raw_path`.
        #     data = Data(...)
        #
        #     if self.pre_filter is not None and not self.pre_filter(data):
        #         continue
        #
        #     if self.pre_transform is not None:
        #         data = self.pre_transform(data)
        #
        #     torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(i)))
        #     i += 1
        pass

    def len(self):
        """
        返回数据集长度（未实现）
        """
        pass

    def __len__(self):
        return self.len()

    def get(self, idx):
        """
        获取指定索引的数据（未实现）
        """
        pass

    @staticmethod
    def get_data(idx, k):
        """
        静态方法：根据索引和kernel名称加载数据

        参数:
            idx: 数据索引
            k: kernel名称

        返回:
            data: 图数据（PyTorch Geometric Data对象）
            code_content: 代码内容（从 .c 文件读取的字符串）
        """
        # 根据kernel所属的benchmark确定路径
        if k in MACHSUITE_KERNEL:
            gt = join(graph_path, 'machsuite', k)
            ct = join(code_path, 'machsuite', k)
        else:
            gt = join(graph_path, 'poly', k)
            ct = join(code_path, 'poly', k)
        # 加载图数据
        data = torch.load(osp.join(gt, 'data_{}.pt'.format(idx)))
        # 读取 .c 文件内容
        code_file = join(ct, f'{idx}.c')
        with open(code_file, 'r', encoding='utf-8') as f:
            code_content = f.read()
        return data, code_content