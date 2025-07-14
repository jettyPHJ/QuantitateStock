import pandas as pd
import random
import numpy as np

file_path = '机器学习数据源.xlsx'
exclude_companies = ['英伟达','苹果'] # 训练数据中排除的公司
exclude_columns = ['时间','财务年','股票代码'] # 不参与训练的列
min_sample_size = 6 #超参数，但是至少为3

def get_excel_meta(file_path):
    # 读取 Excel 文件
    excel_data = pd.ExcelFile(file_path)

    company_names = excel_data.sheet_names
    
    #检查Excel所有sheet的列名是否一致
    first_sheet = pd.read_excel(excel_data, sheet_name=company_names[0])
    reference_columns = list(first_sheet.columns)

    for sheet in company_names[1:]:
        df = pd.read_excel(excel_data, sheet_name=sheet, nrows=0)  # 只读取列名
        if list(df.columns) != reference_columns:
            raise ValueError(f"列名不一致：'{sheet}' 与 '{company_names[0]}' 的列名不同")

    # 将第一个表单的特征列添加到 feature_columns (一维数组)
    feature_columns = reference_columns

    return company_names, feature_columns

company_names, columns = get_excel_meta(file_path)

# 读取excel数据
def load_excel(file_path, company_names, columns): 
    data_map = {}
    for company_name in (c for c in company_names if c not in exclude_companies):
        df = pd.read_excel(file_path, sheet_name=company_name)
        
        # 提取存在的列
        data_map[company_name] = {}
        for col in columns:
            data_map[company_name][col] = df[col].to_numpy()
    return data_map
load_excel(file_path, company_names, columns)

#TODO：生成归一化后的财务数据和新闻的特征向量
def generate_synthetic_data(data_map:dict):
    """生成训练的财务数据"""
    print('公司：',company_names)  # 输出公司名列表
    print('指标：')  # 输出每个公司对应的特征列
    data = []
    
    for _, company_data in data_map.items():
        row_count = len(next(iter(company_data.values())))
        
        # 至少要足够数据才能构造训练样本
        if row_count < min_sample_size:
            continue
        
        # 每个公司生成的样本数量
        sample_size = int(row_count // 1.5)
        
        feature_names = list(company_data.keys())

        for _ in range(sample_size):
            # 随机决定子序列长度
            seq_len = random.randint(min_sample_size-1, row_count-1)
            
            # 随机选择起点，使得子序列不包含最后一行
            max_start = row_count - seq_len - 1
            start_idx = random.randint(0, max_start)

            # 提取每一列在指定范围的数据
            raw_data = []
            sample = []
            target = 0.0
            for col_name in feature_names:               
                # 提取子序列
                col_data = company_data[col_name][start_idx : start_idx + seq_len]
                col_arr = np.array(col_data, dtype=float) 
                # 如果是目标列 '平均股价'，在同样的 min-max 体系下计算 target
                if col_name == '平均股价':
                    target = company_data[col_name][start_idx + seq_len]
                # 对原始数据使用 Sigmoid 归一化
                raw_data.append(col_data)
                col_norm = sigmoid_normalize(col_arr) 
                sample.append(col_norm)
             # 组装 [seq_len, feature_dim]
            orin_data = np.stack(raw_data, axis=1)
            finacial_features = np.stack(sample, axis=1)
            news_features = []
            data.append({
                'origin': orin_data,
                'finacial_features': finacial_features,
                'news_features':news_features,
                'target': target
            })
            
    return data

def sigmoid_normalize(arr, scale=10.0):
    arr = np.array(arr, dtype=float)
    min_val, max_val = np.min(arr), np.max(arr)
    scaled = (arr - min_val) / (max_val - min_val + 1e-8)
    centered = (scaled - 0.5) * scale
    return 1 / (1 + np.exp(-centered))

def get_index(feature_name):
    """获取特征名称的索引"""
    try:
        return columns.index(feature_name)
    except ValueError:
        raise ValueError(f"特征名称 '{feature_name}' 不在特征列中。")