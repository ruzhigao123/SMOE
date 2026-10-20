# import torch
#
#
# file_path = '/home/rzgao/idea2/MPM-Innovation-master/MPM-Innovation-master/two_tower_datasets/code/ast_final_dataset/machsuite/nw/15.ast.pt'  # 换成实际文件名
# # file_path = '/home/rzgao/idea2/MPM-Innovation-master/MPM-Innovation-master/two_tower_dataset/graph/machsuite/nw/data_1.pt'  # 换成实际文件名
#
#
# def print_x_and_labels(path):
#     # 1. 加载数据
#     try:
#         data = torch.load(path, map_location='cpu')
#     except Exception as e:
#         print(f"加载失败: {e}")
#         return
#
#     print(f"=== 深入检查文件: {path} ===")
#
#     if hasattr(data, 'x'):
#         x = data.x
#         # 3. 打印前几行看看数值（每行打印前10个特征）
#         print("\n3. 特征数值预览 (前3个节点的前10维特征):")
#         # 限制打印列数，防止刷屏
#         preview_rows = min(3, x.shape[0])
#         preview_cols = min(10, x.shape[1])
#         for i in range(preview_rows):
#             print(f"   节点 {i}: {x[i, :preview_cols].tolist()} ...")
#
#
#     # 2. 检查节点特征 x
#     if hasattr(data, 'x'):
#         x = data.x
#         print(f"\n[1. 节点特征 x]")
#         print(f"   形状: {x.shape} (节点数: {x.shape[0]}, 维度: {x.shape[1]})")
#
#         # 对应图片中的 10 个特征
#         struct_feat_names = [
#             "is_pragma", "II", "factor", "unroll", "tile",
#             "partition", "depth", "text length", "is_loop", "is_control"
#         ]
#
#         # if x.shape[1] == 778:
#         #     print(f"   最后 10 位结构特征预览 (对应图片列表):")
#         #     last_10 = x[0, -10:].tolist()
#         #     for name, val in zip(struct_feat_names, last_10):
#         #         print(f"      - {name.ljust(12)}: {val}")
#         if x.shape[1] == 778:
#             # 1. 提取所有节点的硬件特征部分 (最后 10 列)
#             hw_features = x[:, -10:]
#
#             # 2. 找到 is_pragma == 1 的节点索引
#             # 根据你的 manual_feat 定义，第一位是 is_pragma
#             pragma_mask = (hw_features[:, 0] == 1)
#             pragma_indices = torch.where(pragma_mask)[0]
#
#             if len(pragma_indices) == 0:
#                 print("🔍 未发现 is_pragma=1 的节点。")
#             else:
#                 print(f"🔥 发现 {len(pragma_indices)} 个 Pragma 相关节点，打印详细硬件特征:")
#
#                 # 3. 遍历这些特定的节点
#                 for idx in pragma_indices:
#                     node_idx = idx.item()
#                     last_10 = hw_features[node_idx].tolist()
#
#                     print(f"  📍 节点 ID: {node_idx} (Pragma 节点)")
#                     for name, val in zip(struct_feat_names, last_10):
#                         # 针对 is_pragma 这种开关型特征打印整数，其他打印浮点数
#                         if name == "is_pragma":
#                             print(f"      - {name.ljust(12)}: {int(val)}")
#                         else:
#                             print(f"      - {name.ljust(12)}: {val:.4f}")
#
#
#     # 3. 打印该配置的所有标签 (Targets)
#     print(f"\n[2. 配置标签/预测目标]")
#     # 获取 data 对象中所有的键
#     all_keys = data.keys() if hasattr(data, 'keys') else data.__dict__.keys()
#
#     # 定义我们关心的常见标签名（排除掉图结构相关的键）
#     exclude_keys = {'x', 'edge_index', 'edge_attr', 'batch', 'ptr', 'num_nodes'}
#
#     found_label = False
#     for key in all_keys:
#         if key in exclude_keys:
#             continue
#
#         val = getattr(data, key) if hasattr(data, key) else data[key]
#         found_label = True
#
#         # 处理不同类型的标签数值
#         if torch.is_tensor(val):
#             # 如果是单数值张量
#             if val.numel() == 1:
#                 print(f"   - {key.ljust(10)}: {val.item()}")
#             # 如果是多维张量（如 y）
#             else:
#                 print(f"   - {key.ljust(10)}: {val.tolist()} (Shape: {list(val.shape)})")
#         else:
#             print(f"   - {key.ljust(10)}: {val}")
#
#     if not found_label:
#         print("   未发现明显的预测标签。")
#
#
# if __name__ == "__main__":
#     print_x_and_labels(file_path)
#
#
import torch


from os.path import abspath, dirname, join
file_path = join(dirname(abspath(__file__)), 'ast_final_dataset_1', 'machsuite', 'nw', '15.ast.pt')  # 换成实际文件名
# file_path = '/home/rzgao/idea2/MPM-Innovation-master/MPM-Innovation-master/two_tower_dataset/graph/machsuite/nw/data_1.pt'  # 换成实际文件名


def print_x_and_labels(path):
    # 1. 加载数据
    try:
        data = torch.load(path, map_location='cpu')
    except Exception as e:
        print(f"加载失败: {e}")
        return

    print(f"=== 深入检查文件: {path} ===")

    if hasattr(data, 'x'):
        x = data.x
        # 3. 打印前几行看看数值（每行打印前10个特征）
        print("\n3. 特征数值预览 (前3个节点的前10维特征):")
        # 限制打印列数，防止刷屏
        preview_rows = min(3, x.shape[0])
        preview_cols = min(10, x.shape[1])
        for i in range(preview_rows):
            print(f"   节点 {i}: {x[i, :preview_cols].tolist()} ...")


    # 2. 检查节点特征 x
    if hasattr(data, 'x'):
        x = data.x
        print(f"\n[1. 节点特征 x]")
        print(f"   形状: {x.shape} (节点数: {x.shape[0]}, 维度: {x.shape[1]})")

        # 与 ast_dataset_upgrade.py 中 _get_advanced_hw_features 保持一致的 16 维定义
        struct_feat_names = [
            "is_pragma_or_accel",          # 0
            "pipeline_ii",                 # 1
            "parallel_or_unroll_factor",   # 2
            "tile_factor",                 # 3
            "pipeline_off_penalty",        # 4
            "has_pipeline",                # 5
            "has_parallel_or_unroll",      # 6
            "has_reduction",               # 7
            "has_flatten",                 # 8
            "has_scop",                    # 9
            "has_partition",               # 10
            "has_interface_or_bus",        # 11
            "is_loop_node",                # 12
            "is_branch_node",              # 13
            "norm_depth",                  # 14
            "norm_text_len",               # 15
        ]

        # if x.shape[1] == 778:
        #     print(f"   最后 10 位结构特征预览 (对应图片列表):")
        #     last_10 = x[0, -10:].tolist()
        #     for name, val in zip(struct_feat_names, last_10):
        #         print(f"      - {name.ljust(12)}: {val}")
        if x.shape[1] >= 16:
            # 1. 提取所有节点的硬件特征部分 (最后 16 列)
            hw_features = x[:, -16:]

            # 2. 找到 is_pragma == 1 的节点索引
            # 根据 ast_dataset_upgrade.py 定义，第一位是 is_pragma_or_accel
            pragma_mask = (hw_features[:, 0] == 1)
            pragma_indices = torch.where(pragma_mask)[0]

            if len(pragma_indices) == 0:
                print("🔍 未发现 is_pragma=1 的节点。")
            else:
                print(f"🔥 发现 {len(pragma_indices)} 个 Pragma 相关节点，打印详细硬件特征:")

                # 3. 遍历这些特定的节点
                for idx in pragma_indices:
                    node_idx = idx.item()
                    last_16 = hw_features[node_idx].tolist()

                    print(f"  📍 节点 ID: {node_idx} (Pragma 节点)")
                    for name, val in zip(struct_feat_names, last_16):
                        # 针对 is_pragma 这种开关型特征打印整数，其他打印浮点数
                        if name == "is_pragma_or_accel":
                            print(f"      - {name.ljust(12)}: {int(val)}")
                        else:
                            print(f"      - {name.ljust(12)}: {val:.4f}")
        else:
            print(f"⚠️ 当前特征维度为 {x.shape[1]}，不足以解析最后16维硬件特征。")


    # 3. 打印该配置的所有标签 (Targets)
    print(f"\n[2. 配置标签/预测目标]")
    # 获取 data 对象中所有的键
    all_keys = data.keys() if hasattr(data, 'keys') else data.__dict__.keys()

    # 定义我们关心的常见标签名（排除掉图结构相关的键）
    exclude_keys = {'x', 'edge_index', 'edge_attr', 'batch', 'ptr', 'num_nodes'}

    found_label = False
    for key in all_keys:
        if key in exclude_keys:
            continue

        val = getattr(data, key) if hasattr(data, key) else data[key]
        found_label = True

        # 处理不同类型的标签数值
        if torch.is_tensor(val):
            # 如果是单数值张量
            if val.numel() == 1:
                print(f"   - {key.ljust(10)}: {val.item()}")
            # 如果是多维张量（如 y）
            else:
                print(f"   - {key.ljust(10)}: {val.tolist()} (Shape: {list(val.shape)})")
        else:
            print(f"   - {key.ljust(10)}: {val}")

    if not found_label:
        print("   未发现明显的预测标签。")


if __name__ == "__main__":
    print_x_and_labels(file_path)


