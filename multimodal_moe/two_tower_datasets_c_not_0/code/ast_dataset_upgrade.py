# import os
# import json
# import torch
# import re
# from torch_geometric.data import Data, Dataset
# from tqdm import tqdm
# from transformers import AutoTokenizer, AutoModel
#
# # ==============================
# # 1. 环境配置与模型加载
# # ==============================
# from os.path import abspath, dirname, join
# CODEBERT_PATH = os.getenv(
#     "CODEBERT_DIR",
#     join(dirname(dirname(dirname(abspath(__file__)))), 'codebert')
# )
# if not os.path.isdir(CODEBERT_PATH):
#     _alt = r'd:/research/code/idea2/MPM-Innovation-master/MPM-Innovation-master/codebert'
#     if os.path.isdir(_alt):
#         CODEBERT_PATH = _alt
#
# # 自动检测 GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"📡 正在使用设备: {device}")
#
# tokenizer = AutoTokenizer.from_pretrained(CODEBERT_PATH)
# model = AutoModel.from_pretrained(CODEBERT_PATH).to(device)
# model.eval()
#
#
# def encode_text(text):
#     """使用 CodeBERT 生成 768 维 embedding"""
#     if not text or not text.strip():
#         return torch.zeros(768).to(device)
#
#     with torch.no_grad():
#         inputs = tokenizer(
#             text,
#             return_tensors="pt",
#             truncation=True,
#             max_length=32
#         ).to(device)
#         outputs = model(**inputs)
#         # mean pooling
#         emb = outputs.last_hidden_state.mean(dim=1).squeeze()
#     return emb
#
#
# # ==============================
# # 2. 数据集类定义
# # ==============================
# class ASTGraphDataset(Dataset):
#     def __init__(self, input_root='ast_output', output_root='ast_processed_pt'):
#         super().__init__(input_root)
#         self.input_root = os.path.abspath(input_root)
#         self.output_root = os.path.abspath(output_root)
#
#         # 统计计数器
#         self.total_nodes_processed = 0
#         self.empty_text_count = 0
#
#         if not os.path.exists(self.output_root):
#             os.makedirs(self.output_root)
#             print(f"📁 已创建输出目录: {self.output_root}")
#
#         self.data_list = self._process_and_mirror_load()
#
#         # 打印最终统计报告
#         self._print_stats()
#
#     def _print_stats(self):
#         print("\n" + "=" * 40)
#         print(f"📊 AST 节点文本属性统计报告")
#         print("-" * 40)
#         print(f"🔹 处理节点总数: {self.total_nodes_processed}")
#         print(f"🔸 空文本节点数: {self.empty_text_count}")
#         if self.total_nodes_processed > 0:
#             ratio = (self.empty_text_count / self.total_nodes_processed) * 100
#             print(f"📉 空文本占比:   {ratio:.2f}%")
#         print("=" * 40 + "\n")
#
#     def _process_and_mirror_load(self):
#         all_files = []
#         for dirpath, _, files in os.walk(self.input_root):
#             for f in files:
#                 if f.endswith('.ast.json'):
#                     all_files.append(os.path.join(dirpath, f))
#
#         if not all_files:
#             print(f"❌ 在 {self.input_root} 中未找到 .ast.json 文件。")
#             return []
#
#         processed_data = []
#
#         for json_path in tqdm(all_files, desc="Mirroring & Processing", unit="file"):
#             # 计算相对路径并在输出端重建目录结构
#             rel_path = os.path.relpath(json_path, self.input_root)
#             pt_path = os.path.join(self.output_root, rel_path.replace('.json', '.pt'))
#             os.makedirs(os.path.dirname(pt_path), exist_ok=True)
#
#             # 如果已经存在 .pt 文件，直接加载并统计节点
#             if os.path.exists(pt_path):
#                 data = torch.load(pt_path)
#                 # 注意：从已保存的 pt 加载时，除非重新遍历 JSON，否则无法统计准确的“空文本数”
#                 # 这里我们假设如果 pt 存在，则跳过详细统计以节省速度
#                 processed_data.append(data)
#                 continue
#
#             # 如果不存在，则处理 JSON
#             try:
#                 with open(json_path, 'r', encoding='utf-8') as fp:
#                     ast = json.load(fp)
#
#                 data = self._ast_to_pyg_data(ast, json_path)
#
#                 if data is not None:
#                     torch.save(data, pt_path)
#                     processed_data.append(data)
#             except Exception as e:
#                 print(f"\n❌ 处理出错 {json_path}: {e}")
#
#         return processed_data
#
#     def _parse_pragma(self, text):
#         if not text or '#pragma' not in text:
#             return {'is_pragma': 0, 'II': 1, 'factor': 1, 'unroll': 1, 'tile': 1, 'partition': 0}
#
#         upper = text.upper()
#         features = {'is_pragma': 1}
#
#         m = re.search(r'II\s*=\s*(\d+)', text, re.IGNORECASE)
#         features['II'] = int(m.group(1)) if m else 1
#
#         m = re.search(r'(UNROLL|PARALLEL|FACTOR)\s*=\s*(\d+)', text, re.IGNORECASE)
#         features['factor'] = int(m.group(2)) if m else 1
#         features['unroll'] = features['factor']
#
#         m = re.search(r'TILE.*?FACTOR\s*=\s*(\d+)', text, re.IGNORECASE)
#         features['tile'] = int(m.group(1)) if m else 1
#         features['partition'] = 1 if 'ARRAY_PARTITION' in upper else 0
#
#         return features
#
#     def _ast_to_pyg_data(self, ast, file_path):
#         nodes = []
#         edges = []
#         node_idx = 0
#
#         def traverse(node, parent_idx=None, depth=0):
#             nonlocal node_idx
#             if not isinstance(node, dict): return
#
#             current_idx = node_idx
#             node_idx += 1
#
#             # --- 统计逻辑 ---
#             self.total_nodes_processed += 1
#             text = node.get('text', '')
#             if text is None or (isinstance(text, str) and not text.strip()):
#                 self.empty_text_count += 1
#                 text = ""  # 统一化
#
#             # 文本特征编码
#             text_emb = encode_text(text).cpu()
#
#             pragma_feat = self._parse_pragma(node.get('pragma_full_text', ''))
#
#             # 合并手动定义的特征
#             manual_feat = torch.tensor([
#                 pragma_feat['is_pragma'],
#                 pragma_feat['II'],
#                 pragma_feat['factor'],
#                 pragma_feat['unroll'],
#                 pragma_feat['tile'],
#                 pragma_feat['partition'],
#                 depth / 50.0,
#                 len(text) / 200.0,
#                 1 if 'for' in node.get('type', '') or 'while' in node.get('type', '') else 0,
#                 1 if node.get('type') in ['if', 'switch'] else 0
#             ], dtype=torch.float)
#
#             nodes.append(torch.cat([text_emb, manual_feat]))
#
#             if parent_idx is not None:
#                 edges.append([parent_idx, current_idx])
#
#             if 'children' in node:
#                 for child in node['children']:
#                     traverse(child, current_idx, depth + 1)
#
#         traverse(ast)
#         if not nodes: return None
#
#         x = torch.stack(nodes)
#         edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0),
#                                                                                                       dtype=torch.long)
#
#         data = Data(x=x, edge_index=edge_index)
#         data.file = os.path.basename(file_path)
#         data.num_nodes = x.size(0)
#         return data
#
#     def len(self):
#         return len(self.data_list)
#
#     def get(self, idx):
#         return self.data_list[idx]
#
#
# # ==============================
# # 3. 执行入口
# # ==============================
# if __name__ == "__main__":
#     # 配置输入与输出
#     INPUT_DIR = 'ast_output'  # 原始 JSON 所在目录
#     OUTPUT_DIR = 'ast_processed_pt'  # 准备存放 PT 文件的目录
#
#     print("🚀 启动 AST 转换引擎 (CodeBERT 版)...")
#     dataset = ASTGraphDataset(input_root=INPUT_DIR, output_root=OUTPUT_DIR)
#
#     if len(dataset) > 0:
#         print(f"✨ 任务完成！处理后的文件保存在: {os.path.abspath(OUTPUT_DIR)}")
#         print(f"💡 示例图节点特征维度: {dataset[0].x.shape}")
#
# ----------------------------------------------------------------------------------------
import os
import json
import torch
import re
import math
from os.path import abspath, dirname, join
from torch_geometric.data import Data, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# ==============================
# 1. 环境配置与模型加载
# ==============================
CODEBERT_PATH = os.getenv(
    "CODEBERT_DIR",
    # code → two_tower_datasets_c_not_0 → multimodal_moe → HierarchicalMoE-master/codebert
    join(dirname(dirname(dirname(dirname(abspath(__file__))))), 'codebert')
)
if not os.path.isdir(CODEBERT_PATH) or not os.path.isfile(join(CODEBERT_PATH, 'config.json')):
    for _alt in (
        join(dirname(dirname(dirname(abspath(__file__)))), 'codebert'),  # 旧路径兜底
        r'/home/rzgao/idea2/HierarchicalMoE-master/codebert',
        r'/home/rzgao/idea2/MPM-Innovation-master/MPM-Innovation-master/codebert',
        r'd:/research/code/idea2/MPM-Innovation-master/MPM-Innovation-master/codebert',
    ):
        if os.path.isdir(_alt) and os.path.isfile(join(_alt, 'config.json')):
            CODEBERT_PATH = _alt
            break

# 自动检测 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📡 正在使用设备: {device}")

tokenizer = AutoTokenizer.from_pretrained(CODEBERT_PATH, local_files_only=True)
model = AutoModel.from_pretrained(CODEBERT_PATH, local_files_only=True).to(device)
model.eval()


def encode_text(text):
    """使用 CodeBERT 生成 768 维 embedding"""
    if not text or not text.strip():
        return torch.zeros(768).to(device)

    with torch.no_grad():
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=64  # 已根据建议从 32 调大至 64
        ).to(device)
        outputs = model(**inputs)
        # mean pooling
        emb = outputs.last_hidden_state.mean(dim=1).squeeze()
    return emb


# ==============================
# 2. 数据集类定义
# ==============================
class ASTGraphDataset(Dataset):
    def __init__(self, input_root='ast_output', output_root='ast_processed_pt'):
        super().__init__(input_root)
        self.input_root = os.path.abspath(input_root)
        self.output_root = os.path.abspath(output_root)

        self.total_nodes_processed = 0
        self.empty_text_count = 0

        if not os.path.exists(self.output_root):
            os.makedirs(self.output_root)
            print(f"📁 已创建输出目录: {self.output_root}")

        self.data_list = self._process_and_mirror_load()
        self._print_stats()

    def _print_stats(self):
        print("\n" + "=" * 40)
        print(f"📊 AST 节点文本属性统计报告")
        print("-" * 40)
        print(f"🔹 处理节点总数: {self.total_nodes_processed}")
        print(f"🔸 空文本节点数: {self.empty_text_count}")
        if self.total_nodes_processed > 0:
            ratio = (self.empty_text_count / self.total_nodes_processed) * 100
            print(f"📉 空文本占比:   {ratio:.2f}%")
        print("=" * 40 + "\n")

    def _process_and_mirror_load(self):
        all_files = []
        for dirpath, _, files in os.walk(self.input_root):
            for f in files:
                if f.endswith('.ast.json'):
                    all_files.append(os.path.join(dirpath, f))

        if not all_files:
            print(f"❌ 在 {self.input_root} 中未找到 .ast.json 文件。")
            return []

        processed_data = []
        for json_path in tqdm(all_files, desc="Mirroring & Processing", unit="file"):
            rel_path = os.path.relpath(json_path, self.input_root)
            pt_path = os.path.join(self.output_root, rel_path.replace('.json', '.pt'))
            os.makedirs(os.path.dirname(pt_path), exist_ok=True)

            if os.path.exists(pt_path):
                processed_data.append(torch.load(pt_path))
                continue

            try:
                with open(json_path, 'r', encoding='utf-8') as fp:
                    ast = json.load(fp)
                data = self._ast_to_pyg_data(ast, json_path)
                if data is not None:
                    torch.save(data, pt_path)
                    processed_data.append(data)
            except Exception as e:
                print(f"\n❌ 处理出错 {json_path}: {e}")
        return processed_data

    def _get_advanced_hw_features(self, node, depth):
        """
        提取 16 维硬件与结构特征
        """
        feats = [0.0] * 16
        node_type = node.get('type', '').lower()
        node_text = node.get('text', '')
        raw_text = node.get('pragma_full_text', '').lower()

        # --- A. 语法结构特征 (12-15) ---
        if 'for' in node_type or 'while' in node_type: feats[12] = 1.0
        if node_type in ['if', 'switch', 'case', 'condition']: feats[13] = 1.0
        feats[14] = min(depth / 60.0, 1.5)  # 归一化深度
        feats[15] = min(len(node_text) / 200.0, 2.0)  # 归一化长度

        # --- B. 硬件指令特征 (0-11) ---
        if "#pragma" in raw_text or "accel" in raw_text:
            feats[0] = 1.0

            if "pipeline" in raw_text:
                if "off" in raw_text:
                    feats[4], feats[1] = -1.0, 15.0  # off 状态惩罚
                else:
                    feats[4] = 1.0
                    m = re.search(r'ii\s*=\s*(\d+)', raw_text)
                    feats[1] = float(m.group(1)) if m else 1.0

            if "parallel" in raw_text or "unroll" in raw_text:
                feats[5] = 1.0
                m = re.search(r'factor\s*=\s*(\d+)', raw_text)
                feats[2] = float(m.group(1)) if m else 1.0

            if "tile" in raw_text:
                m = re.search(r'factor\s*=\s*(\d+)', raw_text)
                feats[3] = float(m.group(1)) if m else 1.0

            if "reduction" in raw_text:        feats[6] = 1.0
            if "flatten" in raw_text:          feats[7] = 1.0
            if "scop" in raw_text:             feats[8] = 1.0
            if "partition" in raw_text:        feats[9] = 1.0
            if "interface" in raw_text or "bus" in raw_text: feats[10] = 1.0
            if "false_dependence" in raw_text: feats[11] = 1.0

        return feats

    def _ast_to_pyg_data(self, ast, file_path):
        nodes, edges = [], []
        node_count = 0

        def traverse(node, parent_idx=None, depth=0):
            nonlocal node_count
            if not isinstance(node, dict): return
            current_idx = node_count
            node_count += 1
            self.total_nodes_processed += 1

            text = node.get('text', '') or ""
            if not text.strip(): self.empty_text_count += 1

            text_emb = encode_text(text).cpu()  # 768维
            hw_feats = self._get_advanced_hw_features(node, depth)  # 16维

            full_feat = torch.cat([text_emb, torch.tensor(hw_feats, dtype=torch.float)])
            nodes.append(full_feat)

            if parent_idx is not None:
                edges.append([parent_idx, current_idx])
            for child in node.get('children', []):
                traverse(child, current_idx, depth + 1)

        traverse(ast)
        if not nodes: return None
        x = torch.stack(nodes)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0),
                                                                                                      dtype=torch.long)
        data = Data(x=x, edge_index=edge_index)
        data.file = os.path.basename(file_path)
        return data

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


if __name__ == "__main__":
    # [not_0] 输入/输出均相对本目录（two_tower_datasets_c_not_0/code）
    _here = dirname(abspath(__file__))
    print(f"[not_0] AST upgrade: {_here}")
    dataset = ASTGraphDataset(
        input_root=join(_here, "ast_output"),
        output_root=join(_here, "ast_processed_pt_1"),
    )
