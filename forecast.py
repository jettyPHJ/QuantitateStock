import torch
from torch.utils.data import DataLoader
import pandas as pd
import os

from model.MambaStock import MambaModel
from data_process.finance_data.database import BlockCode
from data_process.data_set import SingleStockDataset, collate_fn

STOCK_CODE = "NVDA.O"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16

# 加载数据
dataset = SingleStockDataset(stock_code=STOCK_CODE, block_code=BlockCode.NASDAQ_Computer_Index)
if len(dataset) == 0:
    print(f"[Info] 股票 {STOCK_CODE} 无有效样本，无法预测。")
    exit(0)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# 加载模型
model = MambaModel(input_dim=len(dataset.feature_columns))

save_dir = f'./model/{model.__class__.__name__}'
model_path = f"{save_dir}/best_model.pth"

model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
model.to(DEVICE)
model.eval()

# 执行预测
all_records = []
with torch.no_grad():
    for origins, features, _ in loader:
        origins = [o.to(DEVICE) for o in origins]
        features = features.to(DEVICE)
        preds = model(origins, features).squeeze(-1).cpu().numpy()

        for batch_idx, (origin, pred) in enumerate(zip(origins, preds)):
            # 仅记录每个样本的最后一个时间步及其预测值
            last_timestep_data = origin[-1].cpu().numpy()  # 获取最后一个时间步的数据
            row_dict = {
                "样本编号": len(all_records) + 1,
                **{col: last_timestep_data[i] for i, col in enumerate(dataset.feature_columns)}, "预测值": pred
            }
            all_records.append(row_dict)

# 保存为 Excel
df = pd.DataFrame(all_records)
os.makedirs("results", exist_ok=True)
excel_path = f"results/{STOCK_CODE}_prediction.xlsx"
df.to_excel(excel_path, index=False)

print(f"[完成] 预测结果已保存至：{excel_path}")
