import yaml
from typing import Dict, List, Optional
import os

base_dir = os.path.dirname(__file__)
block_code_path = os.path.join(base_dir, "block.yaml")


class BlockItem:

    def __init__(self, name_cn: str, name_en: str, description_en: str, item_id: str, path: str):
        self.name_cn = name_cn
        self.name_en = name_en
        self.description_en = description_en
        self.id = item_id
        self.path = path

    def __repr__(self):
        return f"BlockItem(name_cn='{self.name_cn}', name_en='{self.name_en}', id='{self.id}', path='{self.path}')"


class Block:
    _items: Dict[str, BlockItem] = {}
    _raw_data: dict = {}

    @classmethod
    def load_from_yaml(cls, yaml_path: str):
        with open(yaml_path, "r", encoding="utf-8") as f:
            cls._raw_data = yaml.safe_load(f)
        cls._items = {}

        def traverse(d: dict, path: List[str]):
            for k, v in d.items():
                if isinstance(v, dict) and 'id' in v:
                    item_path = " > ".join(path + [k])
                    cls._items[k] = BlockItem(name_cn=k, name_en=v.get('name_en',
                                                                       ''), description_en=v.get('description_en', ''),
                                              item_id=str(v['id']), path=item_path)
                elif isinstance(v, dict):
                    traverse(v, path + [k])

        traverse(cls._raw_data, [])

    @classmethod
    def get_items_by_parent(cls, parent_name: str) -> Dict[str, BlockItem]:
        results = {}
        for item in cls._items.values():
            # 路径拆分成 list
            path_parts = item.path.split(" > ")
            if parent_name in path_parts:
                # 只要 parent 在路径里，就认为它是父节点
                results[item.name_cn] = item
        return results

    @classmethod
    def get(cls, name: str) -> Optional[BlockItem]:
        return cls._items.get(name)

    @classmethod
    def all(cls) -> Dict[str, BlockItem]:
        return cls._items

    @classmethod
    def find_by_code(cls, code: str) -> Optional[BlockItem]:
        for item in cls._items.values():
            if item.id == str(code):
                return item
        return None


Block.load_from_yaml(block_code_path)

# --------------------- 测试入口 (Test Entry Point) ---------------------
if __name__ == "__main__":
    print("----------- 1. Testing Get Item by Name ('石油天然气钻井') -----------")
    item = Block.get("石油天然气钻井")
    if item:
        print(f"  中文名 (name_cn): {item.name_cn}")
        print(f"  英文名 (name_en): {item.name_en}")
        print(f"  ID (id): {item.id}")
        print(f"  完整路径 (path): {item.path}")
        print(f"  英文描述 (description_en): {item.description_en}")
    else:
        print("  未找到指定名称的板块：石油天然气钻井")

    print("\n----------- 2. Testing Get Top-Level Item ('芯片') -----------")
    chip_item = Block.get("芯片")
    if chip_item:
        print(f"  中文名 (name_cn): {chip_item.name_cn}")
        print(f"  英文名 (name_en): {chip_item.name_en}")
        print(f"  ID (id): {chip_item.id}")
        print(f"  完整路径 (path): {chip_item.path}")
    else:
        print("  未找到指定名称的板块：芯片")

    print("\n----------- 3. Testing Find Item by Code ('1000015087000000') -----------")
    found_item = Block.find_by_code("1000015087000000")
    if found_item:
        print(f"  找到项目: {found_item.path}")
        print(f"  对象详情: {found_item}")
    else:
        print("  未找到该ID对应的项目")

    print("\n----------- 4. Iterating All Sub-Items under '能源设备与服务' -----------")
    sub_items = Block.get_items_by_parent("能源设备与服务")
    if sub_items:
        for name, item in sub_items.items():
            print(f"  - {name}: (ID: {item.id}, Path: {item.path})")
    else:
        print("  '能源设备与服务' 下没有找到任何子项目")

    print("\n----------- 5. Iterating All Loaded Items (Total) -----------")
    all_items = Block.all()
    print(f"  总共加载了 {len(all_items)} 个项目。")
    for name, item in all_items.items():
        print(f"  - {name} -> {item.id}")
