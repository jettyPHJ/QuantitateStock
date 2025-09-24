from dataclasses import dataclass, asdict
from typing import List, Dict
from utils.block import Block
import data_source.finance.script.wind as wind
import json
import os
from datetime import datetime
import utils.feature as ft


@dataclass
class BlockCacheItem:
    """缓存中每个板块的信息"""
    id: str
    name: str
    path: str
    codes: List[str]


class BlockConstituentCache:
    """
    用于缓存板块成分股的类。
    通过 JSON 将板块ID映射到 BlockCacheItem。
    """

    def __init__(self, cache_file_path: str = "block_constituents.json"):
        self.cache_file_path = os.path.join(os.path.dirname(__file__), cache_file_path)
        os.makedirs(os.path.dirname(self.cache_file_path), exist_ok=True)
        self._cache_map: Dict[str, BlockCacheItem] = self._load_cache()

    def _load_cache(self) -> Dict[str, BlockCacheItem]:
        """从文件加载缓存，如果文件不存在则返回空字典。"""
        if os.path.exists(self.cache_file_path):
            try:
                with open(self.cache_file_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                    # 将 dict 转成数据类
                    return {k: BlockCacheItem(**v) for k, v in raw.items()}
            except (json.JSONDecodeError, IOError) as e:
                print(f"[Cache Error] 加载缓存文件失败: {e}，将创建一个新的缓存。")
                return {}
        else:
            print("[Cache] 未找到缓存文件，将创建一个新的缓存。")
            return {}

    def _save_cache(self):
        """保存缓存到文件"""
        try:
            # 将 BlockCacheItem 对象转成 dict
            serializable_dict = {block_id: asdict(item) for block_id, item in self._cache_map.items()}

            with open(self.cache_file_path, "w", encoding="utf-8") as f:
                json.dump(serializable_dict, f, ensure_ascii=False, indent=2)

            print(f"[Cache] 缓存已保存到 {self.cache_file_path}")

        except IOError as e:
            print(f"[Cache Error] 保存缓存文件失败: {e}")

    def _fetch_from_api(self, block_code: str) -> List[str]:
        """从Wind API获取板块成分股"""
        print(f"[API] 正在从API获取板块 {block_code} 的成分股...")
        try:
            wset_result = wind.check_wind_data(
                wind.w.wset("sectorconstituent", f"date={datetime.now().date()};sectorid={block_code}"),
                context=f"获取 {block_code} 板块股票")

            if wset_result.ErrorCode == 0 and len(wset_result.Data) > 0:
                codes = ft.build_translated_data_map(wset_result.Fields, wset_result.Data)["wind_code"]
                # 确保是列表
                if isinstance(codes, str):
                    codes = [codes]
                return codes
            return []
        except Exception as e:
            print(f"[Error] 获取板块 {block_code} 股票列表失败: {e}")
            return []

    def get_stock_codes(self, block_code: str, block_name: str = None, force_refresh: bool = False) -> BlockCacheItem:
        """
        获取指定板块的股票代码，返回 BlockCacheItem
        """
        if not force_refresh and block_code in self._cache_map:
            print(f"[Cache] 命中缓存！板块 {block_code} 的成分股已加载。")
            return self._cache_map[block_code]

        codes = self._fetch_from_api(block_code)
        if codes:
            block_item = Block.find_by_code(block_code)
            item = BlockCacheItem(id=block_code, name=block_name if block_name else
                                  (block_item.name_cn if block_item else ""),
                                  path=block_item.path if block_item else "", codes=codes)
            self._cache_map[block_code] = item
            self._save_cache()
        else:
            # 返回空 BlockCacheItem 避免 None
            item = BlockCacheItem(id=block_code, name=block_name or "", path="", codes=[])
            self._cache_map[block_code] = item

        return item

    def find_blocks_by_stock(self, stock_code: str, parent_block_name: str = None) -> List[BlockCacheItem]:
        """
        根据股票代码查找所属板块，返回 BlockCacheItem 列表
        """
        results: List[BlockCacheItem] = []
        for item in self._cache_map.values():
            if stock_code in item.codes:
                if parent_block_name is None or parent_block_name in item.path:
                    results.append(item)
        return results

    def update_all_blocks(self, force_refresh: bool = False):
        """遍历更新所有板块"""
        for name, block_item in Block.all().items():
            print(f"[Update] 更新板块: {name} (ID: {block_item.id})")
            self.get_stock_codes(block_item.id, block_name=block_item.name_cn, force_refresh=force_refresh)
        print("[Update] 所有板块更新完成。")

    def update_blocks_by_parent(self, parent_block_name: str, force_refresh: bool = False):
        """遍历指定父板块下的子板块更新"""
        blocks = Block.get_items_by_parent(parent_block_name)
        if not blocks:
            print(f"[Update] 未找到父板块 '{parent_block_name}' 下的任何子板块。")
            return
        for name, block_item in blocks.items():
            print(f"[Update] 更新板块: {name} (ID: {block_item.id}, Path: {block_item.path})")
            self.get_stock_codes(block_item.id, block_name=block_item.name_cn, force_refresh=force_refresh)
        print(f"[Update] 父板块 '{parent_block_name}' 下所有子板块更新完成。")


block_cache = BlockConstituentCache()

# --------------------- 测试入口 ---------------------
if __name__ == "__main__":
    # # ----------------- 测试 1: 正向获取 -----------------
    print("\n----------- 测试: 获取 '半导体产品' 板块成分股 -----------")
    target_block = Block.get("半导体产品")

    if target_block:
        # 返回 BlockCacheItem
        block_item: BlockCacheItem = block_cache.get_stock_codes(target_block.id)
        print(f"板块名称: {block_item.name}")
        print(f"板块路径: {block_item.path}")
        print(f"股票总数: {len(block_item.codes)}")
        print("前5只股票:", block_item.codes[:5])

        # ----------------- 测试 2: 反向查找 -----------------
        print("\n----------- 测试: 股票反向查找所属板块 -----------")
        if block_item.codes:
            stock_to_check = block_item.codes[0]
            blocks: List[BlockCacheItem] = block_cache.find_blocks_by_stock(stock_to_check,
                                                                            parent_block_name="SP500_WIND行业类")
            print(f"股票 {stock_to_check} 属于以下板块：")
            for b in blocks:
                print(f" - {b.name} (ID: {b.id}, Path: {b.path}, 股票数: {len(b.codes)})")
    else:
        print("未找到 '半导体产品' 板块。")

    # ----------------- 测试 3: 全量更新 -----------------
    # print("\n----------- 测试: 全量更新所有板块 -----------")
    # block_cache.update_all_blocks(force_refresh=True)

    # ----------------- 测试 4: 父板块范围更新 -----------------
    print("\n----------- 测试: 更新 'SP500_WIND行业类' 下所有子板块 -----------")
    block_cache.update_blocks_by_parent("SP500_WIND行业类", force_refresh=True)
