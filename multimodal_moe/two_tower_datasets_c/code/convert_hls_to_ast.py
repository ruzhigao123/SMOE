
# ---- multimodal_moe path bootstrap ----
import os
from os.path import abspath, dirname, join
_CODE_DIR = dirname(abspath(__file__))
_MM_ROOT = dirname(dirname(_CODE_DIR))  # multimodal_moe
_TT_ROOT = dirname(_CODE_DIR)           # two_tower_datasets_c
_HMOE_ROOT = dirname(_MM_ROOT)

import os
import json
from tree_sitter_languages import get_parser
from tqdm import tqdm

parser = get_parser("cpp")

# ====================== 输出到独立目录（不覆盖原文件） ======================
output_root = join(_CODE_DIR, 'ast_output')
os.makedirs(output_root, exist_ok=True)


def extract_hls_features(text):
    features = {}
    if "#pragma" in text or "pragma HLS" in text or "pragma ACCEL" in text:
        features["is_pragma"] = True
        features["pragma_full_text"] = text.strip()  # 关键：完整 pragma 文本
        upper = text.upper()
        if "PIPELINE" in upper:
            features["pragma_type"] = "PIPELINE"
        elif "UNROLL" in upper:
            features["pragma_type"] = "UNROLL"
        elif "ARRAY_PARTITION" in upper:
            features["pragma_type"] = "ARRAY_PARTITION"
        elif "TILE" in upper or "FACTOR" in upper:
            features["pragma_type"] = "TILE"
    if "ap_uint" in text or "ap_int" in text:
        features["has_ap_type"] = True
    return features


def node_to_dict(node, code_bytes, node_id=0, parent_id=None):
    text = code_bytes[node.start_byte:node.end_byte].decode('utf-8', errors='ignore')
    node_dict = {
        "id": node_id,
        "type": node.type,
        "text": text.strip(),
        "start_line": node.start_point[0] + 1,
        "parent_id": parent_id,
        **extract_hls_features(text)
    }
    children = []
    current_id = node_id + 1
    for child in node.children:
        child_dict, next_id = node_to_dict(child, code_bytes, current_id, node_id)
        children.append(child_dict)
        current_id = next_id
    if children:
        node_dict["children"] = children
    return node_dict, current_id


def optimize_ast_dict(node_dict):
    if 'children' not in node_dict:
        return node_dict if node_dict['type'] not in ['comment', 'empty_statement'] else None
    new_children = []
    for child in node_dict['children']:
        optimized_child = optimize_ast_dict(child)
        if optimized_child:
            new_children.append(optimized_child)
    if len(new_children) == 1 and node_dict['type'] in ['compound_statement', 'statement', 'declaration',
                                                        'translation_unit']:
        return new_children[0]
    if not new_children and node_dict['type'] in ['comment', 'empty_statement']:
        return None
    node_dict['children'] = new_children
    return node_dict


root_dir = _CODE_DIR  # gen_dataset_c 写出的 .c 在 poly/machsuite 下
print(f"🚀 开始生成 AST（pragma_full_text 已加入）→ 输出目录: {output_root}")

for dirpath, dirnames, filenames in os.walk(root_dir):
    # 不递归进入 AST 产物目录
    dirnames[:] = [d for d in dirnames if d not in ('ast_output', 'ast_processed_pt_1', 'ast_final_dataset_1', '__pycache__')]
    c_files = [f for f in filenames if f.endswith(".c")]
    if not c_files: continue
    print(f"📂 处理目录: {dirpath} ({len(c_files)} 个文件)")

    for filename in tqdm(c_files, desc=dirpath):
        filepath = os.path.join(dirpath, filename)
        try:
            with open(filepath, "rb") as f:
                code_bytes = f.read()

            tree = parser.parse(code_bytes)
            root = tree.root_node

            main_node = root
            for node in root.children:
                if node.type == "function_definition":
                    decl = node.child_by_field_name("declarator")
                    if decl:
                        func_name = code_bytes[decl.start_byte:decl.end_byte].decode(errors='ignore')
                        if any(x in func_name for x in ["aes256_encrypt_ecb", "kernel_", "top_module", "main"]):
                            main_node = node
                            break

            ast_dict, _ = node_to_dict(main_node, code_bytes)
            optimized_ast_dict = optimize_ast_dict(ast_dict)

            # 输出到 ast_output 目录
            rel_path = os.path.relpath(filepath, root_dir)
            new_path = os.path.join(output_root, rel_path)
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            ast_path = new_path.replace(".c", ".ast.json")

            with open(ast_path, "w", encoding="utf-8") as f:
                json.dump(optimized_ast_dict, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"⚠️ 出错 {filepath} → {str(e)[:200]}")

print("\n🎉 生成完成！")
print(f"所有 .ast.json 已保存到: {os.path.abspath(output_root)}")
print("原目录下的 .c 文件完全没有被修改！")