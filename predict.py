import torch
from torch.utils.data import DataLoader
import pandas as pd
import os
import numpy as np
from data_process.data_set import SingleStockDataset, collate_fn
from model import MambaModel, LSTMAttentionModel
from data_source.finance.script.block import Block


def run_prediction(model_cls, stock_code, block_code, use_finetune_weights=True):
    """
    运行预测流程，支持选择是否加载微调权重。

    :param model_cls: 模型类，例如 MambaStock.MambaModel
    :param stock_code: 要预测的单支股票代码
    :param block_code: 股票所属板块
    :param use_finetune_weights: 是否加载微调权重
    :param result_dir: Excel 输出目录
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"==> 开始预测: 股票代码={stock_code}, 使用设备={device}, 是否加载微调权重={use_finetune_weights}")

    # 配置路径
    save_dir = f'./model/training_artifacts/{model_cls.__name__}'
    model_path = f"{save_dir}/model_finetune.pth" if use_finetune_weights else f"{save_dir}/model.pth"
    result_dir = f'results/{model_cls.__name__}'

    # 加载数据
    dataset = SingleStockDataset(stock_code=stock_code, block_code=block_code)
    if len(dataset) == 0:
        print(f"[Info] 股票 {stock_code} 无有效样本，无法预测。")
        return

    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # 加载模型
    model = model_cls(input_dim=len(dataset.feature_columns)).to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到模型权重文件: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # 执行预测
    all_records = []
    MAPE_list = []
    start = False
    pre_price = 0
    last_row = None  # 缓存上一个样本

    with torch.no_grad():
        for origins, features, _ in loader:
            origins = [o.to(device) for o in origins]
            features = features.to(device)
            preds = model(origins, features).squeeze(-1).cpu().numpy()

            for _, (origin_tensor, pred) in enumerate(zip(origins, preds)):
                last_timestep_tensor = origin_tensor[-1]
                last_timestep_data = last_timestep_tensor.cpu().numpy()

                row_dict = {
                    "样本编号": len(all_records) + 1,
                    **{col: last_timestep_data[i] for i, col in enumerate(dataset.feature_columns)}
                }

                if "区间日均收盘价" in dataset.feature_columns:
                    price_idx = dataset.feature_columns.index("区间日均收盘价")
                    base_price = last_timestep_data[price_idx]
                    current_pre_price = base_price * (1 + pred)
                    row_dict["预测股价"] = current_pre_price

                    if start:
                        mape = abs(pre_price - base_price) / pre_price if pre_price != 0 else float("nan")
                        MAPE_list.append(mape)
                        if last_row is not None:
                            last_row["MAPE"] = mape
                    else:
                        start = True
                    pre_price = current_pre_price
                else:
                    row_dict["预测股价"] = float("nan")
                    # 不需要写 MAPE，保留为 NaN 即可

                # 上一行入表
                if last_row is not None:
                    all_records.append(last_row)

                # 当前行变为上一行，暂存
                last_row = row_dict

    # 循环结束后，别忘了把最后一行也加进去
    if last_row is not None:
        last_row["MAPE"] = float("nan")  # 最后一行无法计算 MAPE
        all_records.append(last_row)

    # 结果汇总
    final_mape = sum(MAPE_list) / len(MAPE_list) if MAPE_list else float("nan")
    deviation_max = max(MAPE_list) if MAPE_list else float("nan")
    # 结果汇总
    if MAPE_list:
        # 创建一个 NumPy 数组的副本，以免修改原始列表
        mape_array = np.array(MAPE_list)

        # 获取排序后的数组
        sorted_mape = np.sort(mape_array)

        # 计算后20%的起始索引
        start_index = int(len(sorted_mape) * 0.8)

        # 获取后20%的误差值
        top_20_percent_mape = sorted_mape[start_index:]

        # 计算后20%的平均误差
        avg_top_20_percent_mape = np.mean(top_20_percent_mape)
    else:
        avg_top_20_percent_mape = float("nan")

    print(f"📊 最终平均 MAPE: {final_mape:.4f} | 最大误差: {deviation_max:.4f} | 前20%最大误差的平均数：{avg_top_20_percent_mape:.4f}")

    # 保存为 Excel
    os.makedirs(result_dir, exist_ok=True)
    excel_path = os.path.join(result_dir, f"{stock_code}_pre.xlsx")
    pd.DataFrame(all_records).to_excel(excel_path, index=False)

    print(f"[完成] 预测结果已保存至：{excel_path}")


# --------------------- 使用入口 ---------------------
if __name__ == "__main__":
    run_prediction(
        model_cls=LSTMAttentionModel,
        stock_code="AMD.O",
        block_code=Block.find_blocks_by_stock("AMD.O", "纳斯达克计算机指数")[0],
        use_finetune_weights=False  # 切换微调 or 预训练模型
    )
