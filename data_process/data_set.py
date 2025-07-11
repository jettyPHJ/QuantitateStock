import pandas as pd
import random
import numpy as np

file_path = '机器学习数据源.xlsx'
exclude_companies = ['英伟达','苹果']
exclude_columns = ['时间','财务年']
min_sample_size = 6 #超参数，但是至少为3

def get_excel_meta(file_path, exclude_companies = [], exclude_columns = []):
    if exclude_columns is None:
        exclude_columns = []

    # 读取 Excel 文件
    excel_data = pd.ExcelFile(file_path)

    company_names = excel_data.sheet_names

    # 读取第一个表单的数据
    df = excel_data.parse(company_names[0])

    # 获取第一个表的列名
    columns = df.columns.tolist()

    # 清理列名：去掉末尾的空白字符
    cleaned_columns = []

    for col in columns:
        col = col.strip()  # 去掉前后空格
        if not col:  # 如果遇到空白列名，停止处理
            break
        if "Unnamed" in col:  # 检查列名是否包含“Unnamed”
            break
        cleaned_columns.append(col)

    # 过滤掉需要排除的公司和列
    company_names = [name for name in company_names if name not in exclude_companies]
    columns_to_include = [col for col in cleaned_columns if col not in exclude_columns]

    # 将第一个表单的特征列添加到 feature_columns (一维数组)
    feature_columns = columns_to_include

    return company_names, feature_columns

company_names, feature_columns = get_excel_meta(file_path, exclude_companies, exclude_columns) #获取公司和评价指标

# 读取excel数据
def load_excel(file_path, company_names, feature_columns): 
    data_map = {}
    for company_name in company_names:
        df = pd.read_excel(file_path, sheet_name=company_name)
        
        # 提取存在的列
        data_map[company_name] = {}
        for col in feature_columns:
            if col in df.columns:
                data_map[company_name][col] = df[col].to_numpy()
    return data_map

def generate_synthetic_data(source:dict):
    """生成训练的财务数据"""
    print('公司：',company_names)  # 输出公司名列表
    print('指标：',feature_columns)  # 输出每个公司对应的特征列
    data = []
    
    for _, company_data in source.items():
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
            features = np.stack(sample, axis=1)
            data.append({
                'origin': orin_data,
                'features': features,
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
        return feature_columns.index(feature_name)
    except ValueError:
        raise ValueError(f"特征名称 '{feature_name}' 不在特征列中。")