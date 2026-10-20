import os
from os.path import abspath, dirname, join
import torch
from torch_geometric.data import Data
from tqdm import tqdm

# ==============================
# 路径配置
# ==============================
# 你之前生成的只有图结构的目录
AST_STRUCT_ROOT = join(dirname(abspath(__file__)), 'ast_processed_pt_1')
# 存放真实标签的原始目录
LABEL_DATA_ROOT = join(dirname(dirname(abspath(__file__))), 'graph')  # two_tower_datasets_c/graph
# 合并后用于训练的最终目录
FINAL_DATASET_ROOT = join(dirname(abspath(__file__)), 'ast_final_dataset_1')


def merge_labels():
    if not os.path.exists(FINAL_DATASET_ROOT):
        os.makedirs(FINAL_DATASET_ROOT)

    # 1. 扫描所有已生成的 AST 结构文件
    all_ast_files = []
    for dirpath, _, files in os.walk(AST_STRUCT_ROOT):
        for f in files:
            if f.endswith('.ast.pt'):
                all_ast_files.append(os.path.join(dirpath, f))

    print(f"🔍 找到 {len(all_ast_files)} 个结构文件，开始匹配标签...")

    # 参考 gen_dataset.py 需要提取的字段
    target_fields = [
        'perf', 'actual_perf', 'quality',
        'util_BRAM', 'util_DSP', 'util_LUT', 'util_FF',
        'total_BRAM', 'total_DSP', 'total_LUT', 'total_FF'
    ]

    success_count = 0
    fail_count = 0

    for ast_path in tqdm(all_ast_files, desc="Merging"):
        try:
            # 加载 CodeBERT 结构数据 (x: [N, 778], edge_index)
            ast_data = torch.load(ast_path)

            # 解析路径获取 benchmark, kernel 和 index
            # 例如: ast_processed_pt/poly/aes/0.ast.pt
            rel_path = os.path.relpath(ast_path, AST_STRUCT_ROOT)
            path_parts = rel_path.split(os.sep)

            benchmark = path_parts[0]  # poly
            kernel_name = path_parts[1]  # aes
            file_name = path_parts[2]  # 0.ast.pt

            # 提取数字索引 (0.ast.pt -> 0)
            idx = file_name.split('.')[0]

            # 构造对应的标签文件路径: graph/poly/aes/data_0.pt
            label_file_path = os.path.join(LABEL_DATA_ROOT, benchmark, kernel_name, f"data_{idx}.pt")

            if not os.path.exists(label_file_path):
                fail_count += 1
                continue

            # 加载含有真实标签的 Data 对象
            label_data = torch.load(label_file_path)

            # --- 核心注入操作 ---
            for field in target_fields:
                if hasattr(label_data, field):
                    # 从标签文件取值，赋给 AST 结构对象
                    val = getattr(label_data, field)
                    setattr(ast_data, field, val)

            # 额外补充元数据，方便 model.py 处理
            ast_data.kernel = kernel_name
            ast_data.benchmark = benchmark
            # model.py 中通常预测 perf，将其设为标准 y 标签
            if hasattr(ast_data, 'perf'):
                ast_data.y = ast_data.perf

                # 保存到最终目录，保持结构
            final_save_path = os.path.join(FINAL_DATASET_ROOT, rel_path)
            os.makedirs(os.path.dirname(final_save_path), exist_ok=True)
            torch.save(ast_data, final_save_path)

            success_count += 1

        except Exception as e:
            print(f"❌ 出错 {ast_path}: {e}")
            fail_count += 1

    print(f"\n✅ 处理完成！")
    print(f"成功合并: {success_count} 个 | 失败/缺失: {fail_count} 个")
    print(f"数据保存位置: {os.path.abspath(FINAL_DATASET_ROOT)}")


if __name__ == "__main__":
    merge_labels()