import re
import numpy as np

dataset = "texas"
data_dict = {}


def replace_characters(input_string):
    name_string = input_string.replace(':', '_').replace('/', '^')
    return name_string


def preprocess_html(html_content):
    lines = html_content.split('\n')

    first_empty_line_index = next((index for index, line in enumerate(lines) if line.strip() == ''), None)

    if first_empty_line_index is not None:
        html_content = '\n'.join(lines[first_empty_line_index + 1:])

    html_content = re.sub(r'<[^>]*>', '', html_content)
    html_content = re.sub(r'\s+', ' ', html_content)

    html_content = html_content.strip()

    return html_content


# 打开文件并逐行读取内容
with open('./processed/' + dataset + '.content', 'r') as file:
    lines = file.readlines()
    for line_number, line in enumerate(lines):
        # 分割每一行，以制表符为分隔符
        parts = line.strip().split('\t')
        # 提取链接和类别信息
        link = parts[0]
        category = parts[-1]
        # 将链接和类别信息添加到字典中
        data_dict[link] = category

import os


def case_insensitive_open(directory, filename):
    # 在目录中获取所有文件名，包括大小写
    all_filenames = [os.path.join(directory, f) for f in os.listdir(directory) if
                     os.path.isfile(os.path.join(directory, f))]

    # 找到大小写不敏感的文件名
    matching_filenames = [f for f in all_filenames if filename.lower() == os.path.basename(f).lower()]
    if len(matching_filenames) == 0:
        filename = filename + '^'
        matching_filenames = [f for f in all_filenames if filename.lower() == os.path.basename(f).lower()]
    if len(matching_filenames) == 0:
        filename = filename + '.html'
        matching_filenames = [f for f in all_filenames if filename.lower() == os.path.basename(f).lower()]
    # 打开找到的文件
    if matching_filenames:
        return matching_filenames[0]
    else:
        print(f"could not find {filename}")


dir_h = './webkb/'+dataset

# 创建一个空字典用于存储文件信息
node_info_dict = {}
i = 0
# 遍历 data_dict 中的每一项
for link, details in data_dict.items():
    # 构建文件路径
    d = dir_h
    f = replace_characters(link)
    file_path = case_insensitive_open(d, f)
    if file_path:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            html_content = file.read()
        print("===================================================")
        print(html_content)
        html_content = preprocess_html(html_content)
        print("===================================================")
        print(html_content)

        # 将文件信息添加到字典中
        node_info_dict[link] = {"ID": i, "text": html_content, "category": details}
        i += 1


# 引用关系
data_list = []
with open('./processed/'+dataset+'.cites', 'r', encoding='utf-8') as file:
    for line in file:
        # 使用空格或制表符分割每行数据，并去除两边的空白符
        columns = line.strip().split(' ')
        # 将分割后的数据添加到二维列表中
        data_list.append(columns)

# label编码生成
label=[]
dui={"student":0,"course":1,"project":2,"staff":3,"faculty":4}
for link, info in node_info_dict.items():
    label.append(dui[info['category']])
label = np.array(label, dtype=np.int64)
text_list = [value['text'] for value in node_info_dict.values()]
text_array = np.array(text_list)

# 生成边
edge_index = []

for index, item in enumerate(data_list):
    edge=[0,0]
    if item[0] in node_info_dict.keys() and item[1] in node_info_dict.keys():
        edge[0] = int(node_info_dict[item[0]]['ID'])
        edge[1] = int(node_info_dict[item[1]]['ID'])
        edge_index.append(edge)
    else:
        print(item)

edge_index = np.array(edge_index, dtype=np.int64)
#
edge_index = edge_index.T
label_texts = np.array(["student", "course", "project", "staff", "faculty"])
np.savez(
    'dataset.npz',
    edges=edge_index,
    node_labels=label,
    node_texts=text_array,
    label_texts=label_texts
)