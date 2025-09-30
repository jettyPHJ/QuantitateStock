import yaml
import json
import os
from typing import Dict, List, Optional
from datetime import datetime
import utils.feature as ft


class BlockItem:
    """
    核心板块数据结构。
    存储了从YAML加载的静态定义信息，以及从API动态获取的成分股列表。
    """

    def __init__(self, name_cn: str, name_en: str, description_en: str, item_id: str, path: str):
        self.id = item_id  # 板块ID
        self.path = path  # 板块层级路径
        self.name_cn = name_cn  # 板块中文名
        self.name_en = name_en  # 板块英文名
        self.description_en = description_en  # 板块英文描述
        self.stock_codes: List[str] = []  # 成分股列表（动态加载）

    def __repr__(self):
        """提供一个清晰的对象表示，方便调试。"""
        return (f"BlockItem(name_cn='{self.name_cn}', id='{self.id}', "
                f"path='{self.path}', stock_codes_count={len(self.stock_codes)})")


class Block:
    """
    板块信息统一管理类 (静态类)
    
    使用流程:
    1. 调用 Block.initialize(yaml_path, cache_path) 初始化。
    2. 通过 Block.get(), Block.find_by_id() 等方法查询板块。
    3. 通过 Block.get_stock_codes() 获取成分股。
    """
    _items_by_name: Dict[str, BlockItem] = {}  # 按中文名索引
    _items_by_id: Dict[str, BlockItem] = {}  # 按ID索引
    _cache_path: Optional[str] = None

    @classmethod
    def initialize(cls):
        """初始化板块管理器"""

        cls._yaml_path = os.path.join(os.path.dirname(__file__), "config/block.yaml")
        cls._cache_path = os.path.join(os.path.dirname(__file__), "config/block_constituents.json")

        cls._load_from_yaml(cls._yaml_path)
        cls._load_from_cache()

    @classmethod
    def _load_from_yaml(cls, yaml_path: str):
        """从YAML文件加载板块定义"""
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                raw_data = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"YAML 配置文件未找到: {yaml_path}")

        cls._items_by_name.clear()
        cls._items_by_id.clear()

        def traverse(d: dict, path: List[str]):
            for k, v in d.items():
                if isinstance(v, dict) and 'id' in v:
                    item_path = " > ".join(path + [k])
                    item = BlockItem(name_cn=k, name_en=v.get('name_en', ''),
                                     description_en=v.get('description_en', ''), item_id=str(v['id']), path=item_path)
                    cls._items_by_name[k] = item
                    cls._items_by_id[item.id] = item
                if isinstance(v, dict):
                    child_dict = {ck: cv for ck, cv in v.items() if isinstance(cv, dict)}
                    if child_dict:
                        traverse(child_dict, path + [k])

        traverse(raw_data, [])

    @classmethod
    def _load_from_cache(cls):
        """从JSON缓存加载成分股数据"""
        if not cls._cache_path or not os.path.exists(cls._cache_path):
            return

        try:
            with open(cls._cache_path, "r", encoding="utf-8") as f:
                cached_codes: Dict[str, List[str]] = json.load(f)
            for block_id, codes in cached_codes.items():
                if block_id in cls._items_by_id:
                    cls._items_by_id[block_id].stock_codes = codes
        except (json.JSONDecodeError, IOError):
            pass

    @classmethod
    def _save_to_cache(cls):
        """保存内存中板块成分股到JSON缓存"""
        if not cls._cache_path:
            return
        data_to_save = {item.id: item.stock_codes for item in cls._items_by_id.values() if item.stock_codes}
        try:
            os.makedirs(os.path.dirname(cls._cache_path), exist_ok=True)
            with open(cls._cache_path, "w", encoding="utf-8") as f:
                json.dump(data_to_save, f, ensure_ascii=False, indent=2)
        except IOError:
            pass

    @classmethod
    def _fetch_stock_codes(cls, block_id: str) -> List[str]:
        """从Wind API获取板块成分股"""
        import data_source.finance.script.wind as wind
        try:
            wset_result = wind.check_wind_data(
                wind.w.wset("sectorconstituent", f"date={datetime.now().date()};sectorid={block_id}"),
                context=f"获取 {block_id} 板块股票")
            if wset_result.ErrorCode == 0 and len(wset_result.Data) > 0:
                codes_map = ft.build_translated_data_map(wset_result.Fields, wset_result.Data)
                codes = codes_map.get("wind_code", [])
                return codes if isinstance(codes, list) else [codes]
            return []
        except Exception:
            return []

    # ------------------ 公共接口 ------------------

    @classmethod
    def get(cls, name_cn: str) -> Optional[BlockItem]:
        """按中文名获取板块"""
        if cls._cache_path is None:
            raise RuntimeError("Block 尚未初始化，请先调用 Block.initialize()")
        return cls._items_by_name.get(name_cn)

    @classmethod
    def find_by_id(cls, item_id: str) -> Optional[BlockItem]:
        """按ID获取板块"""
        if cls._cache_path is None:
            raise RuntimeError("Block 尚未初始化，请先调用 Block.initialize()")
        return cls._items_by_id.get(str(item_id))

    @classmethod
    def all(cls) -> List[BlockItem]:
        """获取所有板块"""
        if cls._cache_path is None:
            raise RuntimeError("Block 尚未初始化，请先调用 Block.initialize()")
        return list(cls._items_by_name.values())

    @classmethod
    def get_stock_codes(cls, block_id: str, force_refresh: bool = False) -> List[str]:
        if cls._cache_path is None:
            raise RuntimeError("Block 尚未初始化，请先调用 Block.initialize()")

        item = cls.find_by_id(block_id)
        if not item:
            raise ValueError(f"未找到板块 ID: {block_id}")

        # ✅ 优先用缓存
        if not force_refresh and item.stock_codes:
            return item.stock_codes

        # 否则刷新
        codes = cls._fetch_stock_codes(item.id)
        item.stock_codes = codes
        cls._save_to_cache()
        return item.stock_codes

    @classmethod
    def find_blocks_by_stock(cls, stock_code: str, parent_block_name: Optional[str] = None) -> List[BlockItem]:
        """反向查找股票所属板块"""
        results = []
        for item in cls._items_by_id.values():
            if stock_code in item.stock_codes:
                if parent_block_name is None or parent_block_name in item.path:
                    results.append(item)
        return results

    @classmethod
    def update_all_blocks(cls, force_refresh: bool = False):
        """全量更新所有板块成分股"""
        for item in cls.all():
            cls.get_stock_codes(item.id, force_refresh=force_refresh)

    @classmethod
    def get_items_by_parent(cls, parent_name: str) -> Dict[str, BlockItem]:
        """
        获取指定父板块下的所有子板块。
        """
        if cls._cache_path is None:
            raise RuntimeError("Block 尚未初始化，请先调用 Block.initialize()")

        results = {}
        for item in cls._items_by_name.values():
            # 确保 parent_name 在路径中，并且不是板块自身
            if parent_name in item.path and item.name_cn != parent_name:
                results[item.name_cn] = item
        return results


# 初始化
Block.initialize()

# ------------------ 测试代码 ------------------

if __name__ == "__main__":

    # 按名称获取板块成分股
    target_block = Block.get("半导体产品")
    if target_block:
        stock_list = Block.get_stock_codes(target_block.id)
        print(f"\n 板块 '{target_block.name_cn}' 成分股: {stock_list}")

    # 反向查找股票所属板块
    stock_code = "NVDA.O"
    blocks = Block.find_blocks_by_stock(stock_code, "SP500_WIND行业类")
    if blocks:
        print(f"股票 '{stock_code}' 所属板块:")
        for b in blocks:
            print(f"  - {b.path}")
    else:
        print(f"股票 '{stock_code}' 未找到所属板块")
