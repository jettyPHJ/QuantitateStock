import pandas as pd
import random
import numpy as np
import events

file_path = '机器学习数据源.xlsx'
exclude_companies = ['英伟达','苹果'] # 训练数据中排除的公司
exclude_columns = ['时间','财务年','股票代码'] # 不参与训练的列
min_sample_size = 6 #超参数，但是至少为3

db = events.NewsDBManager("events.db")
time_column = '时间'
stock_code_column = '股票代码'
target_column = '平均股价'

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
    columns = reference_columns

    return company_names, columns

company_names, columns = get_excel_meta(file_path)

# 读取excel数据,生成财务map和新闻map
def load_excel(file_path, company_names, columns): 
    data_map, news_map = {}, {}
    for company_name in (c for c in company_names if c not in exclude_companies):
        df = pd.read_excel(file_path, sheet_name=company_name)
        
        data_map[company_name], news_map[company_name] = {}, {}
        # 生成data_map
        for col in columns:
            ser = df[col]
            if col == time_column:
                ser = pd.to_datetime(ser, errors='coerce')
                # 按季度聚合并格式化成 2005Q1
                ser = ser.dt.to_period('Q').astype(str)
            data_map[company_name][col] = ser.to_numpy()
        
        # 获取公司股票代码
        stock_code = data_map[company_name][stock_code_column][0].strip()
        if not stock_code:
            raise ValueError(f"{company_name}: stock_code 为空字符串")

        # 构建 news_map：按季度获取 embedding
        for time in data_map[company_name][time_column]:
            if isinstance(time, str) and len(time) >= 5:
                year = int(time[:4])
                quarter = time[4:]
                _, embedding = events.get_news_and_embedding(stock_code, quarter, year, db)
                if embedding:  # 只记录成功获取的
                    news_map[company_name][time] = embedding
                else:
                    print(f"未获取到 embedding：{company_name} - {time}")
            else:
                print(f"时间格式非法：{company_name} - {time}")    

    return data_map, news_map

def get_feature_columns(columns):
    """获取参与训练的特征列"""
    feature_columns = [col for col in columns if col not in exclude_columns]
    return feature_columns

#TODO：针对财务数据做归一化，构建训练数据集
def generate_synthetic_data(data_map:dict, news_map:dict):
    """生成训练的财务数据"""
    feature_columns = get_feature_columns(columns)
    print('公司：', company_names)  # 输出公司名列表
    print('特征列：', feature_columns)  # 输出参与训练的特征列
    data = []
    
    for company_name, company_data in data_map.items():
        row_count = len(next(iter(company_data.values())))
        
        # 至少要足够数据才能构造训练样本
        if row_count < min_sample_size:
            continue
        
        # 每个公司生成的样本数量
        sample_size = int(row_count // 1.5)

        for _ in range(sample_size):
            # 随机决定子序列长度
            seq_len = random.randint(min_sample_size-1, row_count-1)
            
            # 随机选择起点，使得子序列不包含最后一行
            max_start = row_count - seq_len - 1
            start_idx = random.randint(0, max_start)

            # 提取财报数据集中每一列在指定范围的数据
            raw_data = []
            sample = []
            target = 0.0
            for col_name in feature_columns:               
                # 提取子序列
                col_data = company_data[col_name][start_idx : start_idx + seq_len]
                col_arr = np.array(col_data, dtype=float) 
                # 如果是目标列，在同样的 min-max 体系下计算 target
                if col_name == target_column:
                    target = company_data[col_name][start_idx + seq_len]
                # 对原始数据使用 Sigmoid 归一化
                raw_data.append(col_data)
                col_norm = sigmoid_normalize(col_arr) 
                sample.append(col_norm)
            # 组装 [seq_len, feature_dim]
            orin_data = np.stack(raw_data, axis=1)
            finacial_features = np.stack(sample, axis=1)

            # 提取前一期的新闻和要预测期的新闻向量
            t1, t2 = company_data[time_column][start_idx + seq_len - 1 : start_idx + seq_len + 1]
            e1, e2 = news_map[company_name][t1], news_map[company_name][t2]
            news_features = [e1, e2]
        
            data.append({
                'origin': orin_data,
                'finacial_features': finacial_features,
                'news_features': news_features,
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

# 测试
# data_map, news_map = load_excel(file_path, company_names, columns)  
# generate_synthetic_data(data_map, news_map)