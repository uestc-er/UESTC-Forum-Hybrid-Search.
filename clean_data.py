import json
import re
import os

# 输入和输出文件路径
INPUT_FILE = "data/posts_data.json"
OUTPUT_FILE = "data/posts_data_cleaned.json"

def clean_text(text: str) -> str:
    """
    清洗文本的核心函数
    """
    if not text:
        return ""
    
    # 1. 去除 Markdown 格式的图片/表情 
    # 匹配模式：![...](...) 
    # 你的例子：![1155](s)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)

    # 2. 去除方括号及其内容 (用户要求的逻辑)
    # 匹配模式：[xxx]
    # 你的例子：[s:123] 或 [img]...[/img] 或 [quote]
    # 注意：这也可能会误删 "[Python教程]" 这样的标题，但在论坛语境下通常利大于弊
    text = re.sub(r'\[.*?\]', '', text)

    # 3. 去除多余的空白字符
    # 把多个空格、换行符合并成一个空格，使文本更紧凑
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

def main():
    # 1. 检查文件是否存在
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 未找到数据文件: {INPUT_FILE}")
        return

    print(f"📖 正在读取 {INPUT_FILE} ...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. 遍历清洗
    print(f"🧹 开始清洗 {len(data)} 条数据...")
    cleaned_count = 0
    
    cleaned_data = []
    for item in data:
        original_content = item.get('content', '')
        new_content = clean_text(original_content)
        
        # 更新内容
        item['content'] = new_content
        cleaned_data.append(item)
        
        # 简单统计一下有变化的数据
        if len(original_content) != len(new_content):
            cleaned_count += 1

    # 3. 保存结果
    # 建议存为新文件，防止误操作覆盖原始数据
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 清洗完成！")
    print(f"   - 共处理: {len(data)} 条")
    print(f"   - 有内容变动: {cleaned_count} 条")
    print(f"   - 结果已保存至: {OUTPUT_FILE}")

    # 打印前3条看看效果
    print("\n🔍 效果预览 (前3条):")
    for i in range(min(3, len(cleaned_data))):
        print(f"--- 帖子 {i+1} ---")
        print(f"标题: {cleaned_data[i]['title']}")
        print(f"内容: {cleaned_data[i]['content'][:100]}...") # 只打印前100字

if __name__ == "__main__":
    main()