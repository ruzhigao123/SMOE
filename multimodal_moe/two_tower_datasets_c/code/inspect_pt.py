import os
import torch
from tqdm import tqdm

# data_path = '/home/rzgao/idea2/MPM-Innovation-master/MPM-Innovation-master/two_tower_datasets/code/ast_final_dataset'
from os.path import abspath, dirname, join
data_path = join(dirname(dirname(abspath(__file__))), 'graph')


def deep_check_pt_files(path):
    print(f"正在深度扫描目录: {path}")
    all_files = []
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith('.pt'):
                all_files.append(os.path.join(root, f))

    error_list = []
    for f_path in tqdm(all_files):
        try:
            data = torch.load(f_path, map_location='cpu')

            # 1. 检查是否存在 x
            if not hasattr(data, 'x') or data.x is None:
                error_list.append(f"{f_path}: 缺失特征 x")
                continue

            x = data.x
            # 2. 检查维度
            if x.ndim != 2:
                error_list.append(f"{f_path}: x 不是 2D 矩阵, 形状为 {x.shape}")
            elif x.shape[1] != 778:
                error_list.append(f"{f_path}: 宽度错误, 形状为 {x.shape}")

            # 3. 检查节点数对齐 (PNA 报错的常见原因)
            if hasattr(data, 'num_nodes'):
                if data.num_nodes != x.shape[0]:
                    error_list.append(f"{f_path}: num_nodes({data.num_nodes}) 与 x行数({x.shape[0]}) 不匹配")

            # 4. 检查是否有非法标量混入
            if x.numel() == 0:
                error_list.append(f"{f_path}: x 是空的")

        except Exception as e:
            error_list.append(f"{f_path}: 加载失败 - {str(e)}")

    print("\n--- 扫描报告 ---")
    if not error_list:
        print("所有文件均符合 [N, 778] 规范。")
    else:
        print(f"发现 {len(error_list)} 个异常文件:")
        for err in error_list:
            print(f"  {err}")


if __name__ == "__main__":
    deep_check_pt_files(data_path)