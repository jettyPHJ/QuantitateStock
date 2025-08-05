import torch
from torch.utils.data import DataLoader
import pandas as pd
import os

from model.MambaStock import MambaModel
from data_process.finance_data.database import BlockCode
from data_process.data_set import SingleStockDataset, collate_fn

stock_code = "AAPL.O"
finetune_flag = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16

# 加载数据
dataset = SingleStockDataset(stock_code=stock_code, block_code=BlockCode.NASDAQ_Computer_Index)
if len(dataset) == 0:
    print(f"[Info] 股票 {stock_code} 无有效样本，无法预测。")
    exit(0)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# 加载模型
model = MambaModel(input_dim=len(dataset.feature_columns))

save_dir = f'./model/{model.__class__.__name__}'
model_path = f"{save_dir}/model.pth"
if finetune_flag:
    model_path = f"{save_dir}/model_finetune.pth"

model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
model.to(DEVICE)
model.eval()

# 执行预测
all_records = []
MAPE_list = []  # 用于存储每个样本的 MAPE 值
start = False
pre_price = 0

with torch.no_grad():
    for origins, features, _ in loader:
        origins = [o.to(DEVICE) for o in origins]
        features = features.to(DEVICE)
        preds = model(origins, features).squeeze(-1).cpu().numpy()

        for batch_idx, (origin_tensor, pred) in enumerate(zip(origins, preds)):
            last_timestep_tensor = origin_tensor[-1]  # 最后一个时间步 [feature_dim]
            last_timestep_data = last_timestep_tensor.cpu().numpy()

            row_dict = {
                "样本编号": len(all_records) + 1,
                **{col: last_timestep_data[i] for i, col in enumerate(dataset.feature_columns)}
            }

            # 预测股价 = 区间日均收盘价 * (1 + pred)
            if "区间日均收盘价" in dataset.feature_columns:
                price_idx = dataset.feature_columns.index("区间日均收盘价")
                base_price = last_timestep_data[price_idx]

                # 当前样本的预测价格
                current_pre_price = base_price * (1 + pred)
                row_dict["预测股价"] = current_pre_price

                if start:
                    if pre_price != 0:
                        mape = abs(pre_price - base_price) / pre_price
                        MAPE_list.append(mape)
                        row_dict["MAPE"] = mape
                    else:
                        row_dict["MAPE"] = float("nan")
                else:
                    start = True
                    row_dict["MAPE"] = float("nan")

                pre_price = current_pre_price

            else:
                row_dict["预测股价"] = float("nan")
                row_dict["MAPE"] = float("nan")

            all_records.append(row_dict)

# 在循环结束后，计算所有样本的平均 MAPE，最大误差
final_mape = sum(MAPE_list) / len(MAPE_list) if MAPE_list else float("nan")
deviation_max = max(MAPE_list) if MAPE_list else float("nan")

print(f" 最终的平均 MAPE 为: {final_mape} \n 最大误差为: {deviation_max}")

# 保存为 Excel
df = pd.DataFrame(all_records)
os.makedirs("results", exist_ok=True)
excel_path = f"results/{stock_code}_prediction.xlsx"
df.to_excel(excel_path, index=False)

print(f"[完成] 预测结果已保存至：{excel_path}")
