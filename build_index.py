#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
索引构建模块
功能：读取JSON数据，分别构建"向量索引"和"关键词索引"
"""

import json
import pickle
import os
import sys
import math
from typing import List, Dict, Any
from tqdm import tqdm # 导入进度条库

# 导入必要的库
try:
    import jieba
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer
    import chromadb
    from chromadb.config import Settings
except ImportError as e:
    print(f"导入库失败: {e}")
    print("请先安装依赖: pip install chromadb sentence-transformers rank_bm25 jieba tqdm")
    sys.exit(1)


class IndexBuilder:
    # 【修改点1】默认路径改为 cleaned 版本
    def __init__(self, data_path: str = "data/posts_data_cleaned.json", 
                 chroma_db_path: str = "chroma_db",
                 embedding_model_name: str = "shibing624/text2vec-base-chinese"):
        self.data_path = data_path
        self.chroma_db_path = chroma_db_path
        self.embedding_model_name = embedding_model_name
        
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        os.makedirs(chroma_db_path, exist_ok=True)
        
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        
    def load_data(self) -> List[Dict[str, Any]]:
        """加载JSON数据"""
        if not os.path.exists(self.data_path):
            print(f"❌ 数据文件不存在: {self.data_path}")
            print("请先运行 clean_data.py 进行数据清洗！")
            return []
        
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"📚 成功加载 {len(data)} 条帖子数据")
            return data
        except Exception as e:
            print(f"加载数据失败: {e}")
            return []
    
    def initialize_models(self):
        """初始化嵌入模型和ChromaDB客户端"""
        print("⏳ 正在初始化嵌入模型 (首次运行会自动下载，约400MB)...")
        try:
            self.embedding_model = SentenceTransformer(
                self.embedding_model_name,
                device='cpu' # 笔记本使用 CPU 即可
            )
            print(f"✅ 嵌入模型加载成功: {self.embedding_model_name}")
        except Exception as e:
            print(f"⚠️ 加载中文模型失败: {e}")
            print("🔄 尝试使用备用模型...")
            self.embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device='cpu')
        
        print("⏳ 正在初始化 ChromaDB...")
        try:
            self.chroma_client = chromadb.PersistentClient(path=self.chroma_db_path)
            print("✅ ChromaDB 客户端就绪")
        except Exception as e:
            print(f"❌ 初始化 ChromaDB 失败: {e}")
            raise
    
    def build_vector_index(self, posts: List[Dict[str, Any]]) -> bool:
        """构建向量索引 (ChromaDB) - 支持分批处理"""
        if not posts: return False
        
        try:
            # 重置 Collection
            try:
                self.chroma_client.delete_collection("forum_posts")
            except:
                pass
            
            self.collection = self.chroma_client.create_collection(
                name="forum_posts",
                metadata={"description": "UESTC Forum Posts"}
            )
            
            # 准备数据
            print("🔄 正在准备向量数据...")
            ids = []
            documents = []
            metadatas = []
            
            for i, post in enumerate(posts):
                # 确保 ID 是字符串
                post_id = str(post.get('id', i)) 
                
                # 组合标题和内容，让语义更丰富
                title = post.get('title', '无标题')
                content = post.get('content', '')
                # 如果正文太短，重复一下标题增强权重
                doc_text = f"{title}\n{content}" if len(content) > 5 else f"{title}\n{title}"
                
                # 长度截断 (Chroma 限制)
                if len(doc_text) > 8000: doc_text = doc_text[:8000]
                
                ids.append(post_id)
                documents.append(doc_text)
                metadatas.append({
                    'title': title,
                    'author': post.get('author', '未知'),
                    'url': post.get('url', ''),
                    'timestamp': str(post.get('timestamp', '')),
                    'id': post_id
                })
            
            # 【修改点2】分批写入 (Batch Processing)
            BATCH_SIZE = 64
            total_batches = math.ceil(len(ids) / BATCH_SIZE)
            
            print(f"🚀 开始向量化并存入数据库 (共 {len(ids)} 条，分 {total_batches} 批)...")
            
            for i in tqdm(range(0, len(ids), BATCH_SIZE), desc="向量化进度"):
                end = i + BATCH_SIZE
                batch_ids = ids[i:end]
                batch_docs = documents[i:end]
                batch_metas = metadatas[i:end]
                
                # 生成向量
                batch_embeddings = self.embedding_model.encode(
                    batch_docs, 
                    normalize_embeddings=True # 归一化向量，这对余弦相似度很重要
                ).tolist()
                
                # 写入 Chroma
                self.collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    documents=batch_docs,
                    metadatas=batch_metas
                )
            
            print(f"✅ 向量索引构建完成！")
            return True
            
        except Exception as e:
            print(f"❌ 构建向量索引失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def build_keyword_index(self, posts: List[Dict[str, Any]]) -> bool:
        """构建关键词索引 (BM25)"""
        if not posts: return False
        
        try:
            print("🏗️ 正在构建关键词索引 (BM25)...")
            
            tokenized_corpus = []
            doc_mapping = [] 
            
            # 使用 tqdm 显示进度
            for post in tqdm(posts, desc="分词进度"):
                # 组合标题和内容
                text = f"{post.get('title', '')} {post.get('content', '')}"
                
                # jieba 分词
                tokens = jieba.lcut_for_search(text)
                
                # 简单的停用词过滤 (过滤掉标点和单字)
                filtered_tokens = [t for t in tokens if len(t.strip()) > 1]
                
                tokenized_corpus.append(filtered_tokens)
                doc_mapping.append(post) # 存下原始数据，方便检索时查阅
            
            # 构建模型
            bm25 = BM25Okapi(tokenized_corpus)
            
            # 保存
            index_data = {
                'bm25_model': bm25,
                'doc_mapping': doc_mapping
            }
            
            output_path = "data/bm25_index.pkl"
            with open(output_path, 'wb') as f:
                pickle.dump(index_data, f)
            
            print(f"✅ 关键词索引已保存: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ 构建关键词索引失败: {e}")
            return False
    
    def build_all_indices(self):
        print("🚀 启动索引构建流程...")
        posts = self.load_data()
        if not posts: return
        
        self.initialize_models()
        
        v_ok = self.build_vector_index(posts)
        k_ok = self.build_keyword_index(posts)
        
        if v_ok and k_ok:
            print("\n🎉🎉🎉 所有索引构建成功！")
            print(f"📂 向量库存放于: {self.chroma_db_path}")
            print(f"📂 BM25 存放于: data/bm25_index.pkl")
        else:
            print("\n⚠️ 即使部分失败，您可能仍可运行搜索，但功能受限。")

if __name__ == "__main__":
    # 确保 data 目录存在
    if not os.path.exists("data"):
        os.makedirs("data")
        
    builder = IndexBuilder()
    builder.build_all_indices()