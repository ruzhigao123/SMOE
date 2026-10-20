import os
import torch
from tqdm import tqdm

# 建议直接指向数据集的最上级目录
from os.path import abspath, dirname, join
data_path = join(dirname(abspath(__file__)), 'ast_final_dataset_1')


def check_all_pt_files(path):
    print(f"正在扫描目录: {path} ...")

    # 获取目录下所有 .pt 文件的完整路径（包括子文件夹）
    all_pt_files = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith('.pt'):
                all_pt_files.append(os.path.join(root, f))

    if not all_pt_files:
        print("错误：未找到任何 .pt 文件！请检查路径是否正确，或目录下是否有 .pt 文件。")
        return

    print(f"找到 {len(all_pt_files)} 个文件，开始检查...")

    dim_counts = {}
    error_files = []

    for f_path in tqdm(all_pt_files):
        try:
            # map_location='cpu' 可以防止 GPU 显存溢出
            data = torch.load(f_path, map_location='cpu')

            # 检查是否有特征矩阵 x
            if hasattr(data, 'x') and data.x is not None:
                dim = data.x.shape[1]
                # 统计维度
                dim_counts[dim] = dim_counts.get(dim, 0) + 1
                if dim != 778:
                    error_files.append((os.path.basename(f_path), dim))
            else:
                print(f"\n文件 {os.path.basename(f_path)} 不包含特征 x")

        except Exception as e:
            # 忽略无法直接加载的对象（有些 .pt 可能是非 Data 对象）
            pass

    print("\n--- 维度统计结果 ---")
    for dim, count in dim_counts.items():
        print(f"维度 {dim}: {count} 个节点/文件分布")

    if error_files:
        print(f"\n--- 异常文件示例 (前5个) ---")
        for fn, dim in error_files[:5]:
            print(f"文件名: {fn}, 实际维度: {dim}")
    else:
        print("\n[结果] 扫描完成。")


if __name__ == "__main__":
    check_all_pt_files(data_path)