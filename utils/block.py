import yaml
from typing import Dict, List
import os

base_dir = os.path.dirname(__file__)
block_code_path = os.path.join(base_dir, "block.yaml")


class BlockItem:

    def __init__(self, name: str, desc: str, code: str):
        self.name = name
        self.desc = desc
        self.code = code

    def __repr__(self):
        return f"BlockItem(name='{self.name}', desc='{self.desc}', code='{self.code}')"


class Block:
    _items: Dict[str, BlockItem] = {}
    _raw_data: dict = {}  # 新增：存储原始YAML结构

    @classmethod
    def load_from_yaml(cls, yaml_path: str):
        with open(yaml_path, "r", encoding="utf-8") as f:
            cls._raw_data = yaml.safe_load(f)  # 记录原始数据
        cls._items = {}

        def traverse(d: dict, path: List[str]):
            for k, v in d.items():
                if isinstance(v, dict):
                    traverse(v, path + [k])
                else:
                    desc = " > ".join(path + [k])
                    name = k
                    cls._items[name] = BlockItem(name=name, desc=desc, code=str(v))

        traverse(cls._raw_data, [])

    @classmethod
    def get_sub_items(cls, entry_key: str) -> Dict[str, BlockItem]:
        """
        只遍历某一级入口（如 SP500_WIND行业类）下的结构，返回对应的 BlockItem 映射。
        """
        result: Dict[str, BlockItem] = {}

        def traverse_subtree(d: dict, path: List[str]):
            for k, v in d.items():
                if isinstance(v, dict):
                    traverse_subtree(v, path + [k])
                else:
                    desc = " > ".join(path + [k])
                    name = k
                    result[name] = BlockItem(name=name, desc=desc, code=str(v))

        subtree = cls._raw_data.get(entry_key)
        if subtree:
            traverse_subtree(subtree, [entry_key])

        return result

    @classmethod
    def get(cls, name: str) -> BlockItem:
        return cls._items.get(name)

    @classmethod
    def all(cls) -> Dict[str, BlockItem]:
        return cls._items

    @classmethod
    def find_by_code(cls, code: str) -> BlockItem:
        for item in cls._items.values():
            if item.code == str(code):
                return item
        return None


Block.load_from_yaml(block_code_path)

# --------------------- 测试入口 ---------------------
if __name__ == "__main__":
    # item = Block.get("石油天然气钻井")
    # if item:
    #     print(item.name)
    #     print(item.desc)
    #     print(item.code)
    # else:
    #     print("未找到指定名称的板块：石油天然气钻井")

    # # 遍历所有
    # for name, item in Block.all().items():
    #     print(name, item.desc, item.code)

    # # 通过代码反查描述
    # found = Block.find_by_code("1000015080000000")
    # print(found.desc if found else "Not found")

    # 遍历子结构
    sub_items = Block.get_sub_items("SP500_WIND行业类")
    for name, item in sub_items.items():
        print(name, item.desc, item.code)
