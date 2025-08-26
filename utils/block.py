import yaml
from typing import Dict, List
import os

base_dir = os.path.dirname(__file__)
block_code_path = os.path.join(base_dir, "tem_block.yaml")


class BlockItem:

    def __init__(self, name: str, desc: str, code: str):
        self.name = name
        self.desc = desc
        self.code = code

    def __repr__(self):
        return f"BlockItem(name='{self.name}', desc='{self.desc}', code='{self.code}')"


class Block:
    _items: Dict[str, BlockItem] = {}

    @classmethod
    def load_from_yaml(cls, yaml_path: str):
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        cls._items = {}

        def traverse(d: dict, path: List[str]):
            for k, v in d.items():
                if isinstance(v, dict):
                    traverse(v, path + [k])
                else:
                    desc = " > ".join(path + [k])
                    name = k
                    cls._items[name] = BlockItem(name=name, desc=desc, code=str(v))

        traverse(data, [])

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
    item = Block.get("石油天然气钻井")
    if item:
        print(item.name)
        print(item.desc)
        print(item.code)
    else:
        print("未找到指定名称的板块：石油天然气钻井")

    # 遍历所有
    for name, item in Block.all().items():
        print(name, item.desc, item.code)

    # 通过代码反查描述
    found = Block.find_by_code("1000015080000000")
    print(found.desc if found else "Not found")
