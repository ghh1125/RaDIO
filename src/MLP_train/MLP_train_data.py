import json

# 读取JSON文件
with open('data_valid.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 处理每条数据
for entry in data:
    original_text = entry['original_text']
    title = entry['title']
    texts = entry['texts']

    # 检查new_entities字段是否存在内容
    if 'new_entities' in entry and entry['new_entities']:
        label = 1  # 如果new_entities字段存在且有内容，则label为1
    else:
        label = 0  # 否则label为0

    # 如果texts字段为空，则使用original_text的内容填充
    if not texts:
        Original_text = "Please introduce the " + title + ":" + original_text
        texts = [Original_text]

    # 更新条目中的字段
    entry['texts'] = texts
    entry['label'] = label

# 保存处理后的数据为MLP_train.json文件
output_data = []
for entry in data:
    output_entry = {
        'original_text': entry['original_text'],
        'title': entry['title'],
        'texts': entry['texts'],
        'label': entry['label']
    }
    output_data.append(output_entry)

with open('MLP_valid.json', 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)
