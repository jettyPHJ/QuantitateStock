import yaml
from typing import Dict, List
import os

base_dir = os.path.dirname(__file__)
block_code_path = os.path.join(base_dir, "block_code.yaml")


class BlockCodeItem:

    def __init__(self, desc: str, code: str):
        self.desc = desc
        self.code = code

    def __repr__(self):
        return f"BlockCodeItem(desc='{self.desc}', code='{self.code}')"


class BlockCode:
    _items: Dict[str, BlockCodeItem] = {}

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
                    cls._items[k] = BlockCodeItem(desc=desc, code=str(v))

        traverse(data, [])

    @classmethod
    def get(cls, name: str) -> BlockCodeItem:
        return cls._items.get(name)

    @classmethod
    def all(cls) -> Dict[str, BlockCodeItem]:
        return cls._items

    @classmethod
    def find_by_code(cls, code: str) -> BlockCodeItem:
        for item in cls._items.values():
            if item.code == str(code):
                return item
        return None


BlockCode.load_from_yaml(block_code_path)

# --------------------- 测试入口 ---------------------
if __name__ == "__main__":
    item = BlockCode.get("石油天然气钻井")
    print(item.desc)
    print(item.code)
    # 遍历所有
    for name, item in BlockCode.all().items():
        print(name, item.desc, item.code)

    # 通过代码反查描述
    found = BlockCode.find_by_code("1000015080000000")
    print(found.desc if found else "Not found")
