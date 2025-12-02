#!/usr/bin/env python3
"""
只构建BM25关键词索引的简化脚本
"""

import json
import pickle
import os
import sys
import jieba
from rank_bm25 import BM25Okapi

def build_bm25_index():
    """只构建BM25关键词索引"""
    print("开始构建BM25关键词索引...")
    
    # 1. 加载数据
    data_file = "data/posts_data.json"
    if not os.path.exists(data_file):
        print(f"数据文件不存在: {data_file}")
        return False
    
    with open(data_file, 'r', encoding='utf-8') as f:
        posts = json.load(f)
    
    print(f"成功加载 {len(posts)} 条帖子数据")
    
    # 2. 准备文档列表
    documents = []
    doc_mapping = []
    
    for i, post in enumerate(posts):
        content = post.get('content', '')
        title = post.get('title', '')
        
        # 合并标题和内容作为文档
        document = f"{title} {content}"
        
        # 使用jieba分词
        tokens = list(jieba.cut_for_search(document))
        
        # 过滤停用词和短词
        filtered_tokens = []
        for token in tokens:
            token = token.strip()
            if len(token) > 1 and not token.isspace():
                filtered_tokens.append(token)
        
        if filtered_tokens:
            documents.append(filtered_tokens)
            doc_mapping.append({
                'id': post.get('id', str(i)),
                'title': title,
                'content': content,
                'author': post.get('author', ''),
                'url': post.get('url', ''),
                'timestamp': post.get('timestamp', '')
            })
    
    if not documents:
        print("没有有效的文档可用于构建BM25索引")
        return False
    
    print(f"准备构建BM25模型，文档数: {len(documents)}")
    
    # 3. 构建BM25模型
    bm25 = BM25Okapi(documents)
    
    # 4. 保存索引和映射
    index_data = {
        'bm25_model': bm25,
        'doc_mapping': doc_mapping,
        'document_count': len(documents)
    }
    
    output_path = "data/bm25_index.pkl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(index_data, f)
    
    print(f"关键词索引构建完成！已保存到: {output_path}")
    print(f"文档总数: {len(documents)}")
    
    # 5. 测试索引
    print("\n测试索引功能...")
    test_queries = ["校园", "学习", "考试"]
    for query in test_queries:
        tokenized_query = list(jieba.cut_for_search(query))
        scores = bm25.get_scores(tokenized_query)
        if len(scores) > 0:
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]
            print(f"查询 '{query}':")
            for idx in top_indices:
                if scores[idx] > 0:
                    print(f"  - {doc_mapping[idx]['title'][:30]}... (分数: {scores[idx]:.4f})")
        else:
            print(f"查询 '{query}': 无匹配结果")
    
    return True

if __name__ == "__main__":
    try:
        success = build_bm25_index()
        if success:
            print("\n✓ BM25关键词索引构建成功！")
            print("下一步: 可以尝试构建向量索引或直接启动服务")
        else:
            print("\n✗ BM25关键词索引构建失败")
            sys.exit(1)
    except Exception as e:
        print(f"构建过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
