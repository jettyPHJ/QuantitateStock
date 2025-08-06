import torch
from torch.utils.data import DataLoader
import pandas as pd
import os
from data_process.data_set import SingleStockDataset, collate_fn
from model import MambaModel, LSTMAttentionModel
from data_process.finance_data.database import BlockCode


def run_prediction(model_cls, stock_code, block_code, use_finetune_weights=True):
    """
    运行预测流程，支持选择是否加载微调权重。

    :param model_cls: 模型类，例如 MambaStock.MambaModel
    :param stock_code: 要预测的单支股票代码
    :param block_code: 股票所属板块 BlockCode
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

    with torch.no_grad():
        for origins, features, _ in loader:
            origins = [o.to(device) for o in origins]
            features = features.to(device)
            preds = model(origins, features).squeeze(-1).cpu().numpy()

            for batch_idx, (origin_tensor, pred) in enumerate(zip(origins, preds)):
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
                        row_dict["MAPE"] = mape
                    else:
                        start = True
                        row_dict["MAPE"] = float("nan")

                    pre_price = current_pre_price
                else:
                    row_dict["预测股价"] = float("nan")
                    row_dict["MAPE"] = float("nan")

                all_records.append(row_dict)

    # 结果汇总
    final_mape = sum(MAPE_list) / len(MAPE_list) if MAPE_list else float("nan")
    deviation_max = max(MAPE_list) if MAPE_list else float("nan")

    print(f"📊 最终平均 MAPE: {final_mape:.4f} | 最大误差: {deviation_max:.4f}")

    # 保存为 Excel
    os.makedirs(result_dir, exist_ok=True)
    excel_path = os.path.join(result_dir, f"{stock_code}_pre.xlsx")
    pd.DataFrame(all_records).to_excel(excel_path, index=False)

    print(f"[完成] 预测结果已保存至：{excel_path}")


# --------------------- 使用入口 ---------------------
if __name__ == "__main__":
    run_prediction(
        model_cls=LSTMAttentionModel,
        stock_code="NVDA.O",
        block_code=BlockCode.NASDAQ_Computer_Index,
        use_finetune_weights=False  # 切换微调 or 预训练模型
    )
