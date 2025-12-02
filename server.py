#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混合检索服务后端 (FastAPI)
功能：提供RESTful API接口，实现混合检索和RRF融合
"""

import json
import pickle
import os
import sys
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

# 导入必要的库
try:
    import jieba
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer
    import chromadb
    from chromadb.config import Settings
    import uvicorn
except ImportError as e:
    print(f"导入库失败: {e}")
    print("请先安装依赖: pip install -r requirements.txt")
    sys.exit(1)


# 数据模型
class SearchRequest(BaseModel):
    """搜索请求模型"""
    query: str
    top_k: int = 20
    fusion_method: str = "rrf"  # rrf, weighted, or simple


class SearchResult(BaseModel):
    """搜索结果模型"""
    id: str
    title: str
    content: str
    author: str
    url: str
    timestamp: str
    score: float
    summary: str


class SearchResponse(BaseModel):
    """搜索响应模型"""
    query: str
    total_results: int
    results: List[SearchResult]
    search_time_ms: float


class HybridSearchEngine:
    """混合搜索引擎核心类"""
    
    def __init__(self, 
                 chroma_db_path: str = "chroma_db",
                 bm25_index_path: str = "data/bm25_index.pkl",
                 embedding_model_name: str = "shibing624/text2vec-base-chinese"):
        """
        初始化搜索引擎
        
        Args:
            chroma_db_path: ChromaDB存储路径
            bm25_index_path: BM25索引文件路径
            embedding_model_name: 嵌入模型名称
        """
        self.chroma_db_path = chroma_db_path
        self.bm25_index_path = bm25_index_path
        self.embedding_model_name = embedding_model_name
        
        # 初始化组件
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        self.bm25_model = None
        self.bm25_doc_mapping = None
        
        # 加载所有组件
        self._initialize_components()
    
    def _initialize_components(self):
        """初始化所有组件"""
        print("正在初始化混合搜索引擎组件...")
        
        # 1. 加载嵌入模型
        print("  加载嵌入模型...")
        try:
            self.embedding_model = SentenceTransformer(
                self.embedding_model_name,
                device='cpu'
            )
            print(f"    嵌入模型加载成功: {self.embedding_model_name}")
        except Exception as e:
            print(f"    加载嵌入模型失败: {e}")
            raise
        
        # 2. 连接ChromaDB
        print("  连接ChromaDB...")
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=self.chroma_db_path,
                settings=Settings(anonymized_telemetry=False)
            )
            self.collection = self.chroma_client.get_collection("forum_posts")
            print(f"    ChromaDB连接成功，文档数: {self.collection.count()}")
        except Exception as e:
            print(f"    连接ChromaDB失败: {e}")
            raise
        
        # 3. 加载BM25索引
        print("  加载BM25索引...")
        try:
            if not os.path.exists(self.bm25_index_path):
                print(f"    BM25索引文件不存在: {self.bm25_index_path}")
                print("    请先运行 build_index.py 构建索引")
                raise FileNotFoundError(f"BM25索引文件不存在: {self.bm25_index_path}")
            
            with open(self.bm25_index_path, 'rb') as f:
                index_data = pickle.load(f)
            
            self.bm25_model = index_data['bm25_model']
            self.bm25_doc_mapping = index_data['doc_mapping']
            print(f"    BM25索引加载成功，文档数: {len(self.bm25_doc_mapping)}")
        except Exception as e:
            print(f"    加载BM25索引失败: {e}")
            raise
        
        print("所有组件初始化完成！")
    
    def _vector_search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        向量搜索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            搜索结果列表
        """
        try:
            # 生成查询向量
            query_embedding = self.embedding_model.encode(
                query,
                convert_to_numpy=True,
                normalize_embeddings=True
            ).tolist()
            
            # 在ChromaDB中搜索
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # 格式化结果
            vector_results = []
            if results['ids'] and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    doc_id = results['ids'][0][i]
                    metadata = results['metadatas'][0][i]
                    distance = results['distances'][0][i]
                    document = results['documents'][0][i] if results['documents'][0] else ""
                    
                    # 将距离转换为相似度分数（距离越小，相似度越高）
                    # ChromaDB使用L2距离，我们将其转换为0-1的相似度分数
                    similarity_score = 1.0 / (1.0 + distance) if distance > 0 else 1.0
                    
                    vector_results.append({
                        'id': doc_id,
                        'title': metadata.get('title', '无标题'),
                        'content': document,
                        'author': metadata.get('author', '未知作者'),
                        'url': metadata.get('url', ''),
                        'timestamp': metadata.get('timestamp', ''),
                        'score': similarity_score,
                        'search_type': 'vector'
                    })
            
            return vector_results
            
        except Exception as e:
            print(f"向量搜索失败: {e}")
            return []
    
    def _keyword_search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        关键词搜索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            搜索结果列表
        """
        try:
            # 对查询进行分词
            tokenized_query = list(jieba.cut_for_search(query))
            
            # 过滤停用词和短词
            filtered_tokens = []
            for token in tokenized_query:
                token = token.strip()
                if len(token) > 1 and not token.isspace():
                    filtered_tokens.append(token)
            
            if not filtered_tokens:
                return []
            
            # 使用BM25进行搜索
            scores = self.bm25_model.get_scores(filtered_tokens)
            
            # 获取top_k个结果的索引
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            # 格式化结果
            keyword_results = []
            for idx in top_indices:
                if idx < len(self.bm25_doc_mapping):
                    doc_info = self.bm25_doc_mapping[idx]
                    score = scores[idx]
                    
                    # BM25分数可能为负数，我们将其归一化到0-1范围
                    normalized_score = max(0.0, min(1.0, (score + 2) / 4))  # 简单归一化
                    
                    keyword_results.append({
                        'id': doc_info.get('id', str(idx)),
                        'title': doc_info.get('title', '无标题'),
                        'content': doc_info.get('content', ''),
                        'author': doc_info.get('author', '未知作者'),
                        'url': doc_info.get('url', ''),
                        'timestamp': doc_info.get('timestamp', ''),
                        'score': normalized_score,
                        'search_type': 'keyword'
                    })
            
            return keyword_results
            
        except Exception as e:
            print(f"关键词搜索失败: {e}")
            return []
    
    def _rrf_fusion(self, vector_results: List[Dict], keyword_results: List[Dict], 
                   top_k: int = 20, k: int = 60) -> List[Dict[str, Any]]:
        """
        倒数排名融合 (Reciprocal Rank Fusion)
        
        Args:
            vector_results: 向量搜索结果
            keyword_results: 关键词搜索结果
            top_k: 最终返回结果数量
            k: RRF参数，通常设为60
            
        Returns:
            融合后的结果列表
        """
        # 创建文档ID到排名的映射
        vector_ranks = {}
        keyword_ranks = {}
        
        # 记录向量搜索结果的排名
        for rank, result in enumerate(vector_results, 1):
            doc_id = result['id']
            vector_ranks[doc_id] = rank
        
        # 记录关键词搜索结果的排名
        for rank, result in enumerate(keyword_results, 1):
            doc_id = result['id']
            keyword_ranks[doc_id] = rank
        
        # 收集所有唯一的文档ID
        all_doc_ids = set(list(vector_ranks.keys()) + list(keyword_ranks.keys()))
        
        # 计算每个文档的RRF分数
        rrf_scores = []
        for doc_id in all_doc_ids:
            vector_rank = vector_ranks.get(doc_id, k + 1)  # 如果未出现，排名设为k+1
            keyword_rank = keyword_ranks.get(doc_id, k + 1)
            
            # RRF公式: score = 1/(k + rank)
            vector_score = 1.0 / (k + vector_rank)
            keyword_score = 1.0 / (k + keyword_rank)
            
            total_score = vector_score + keyword_score
            
            # 获取文档信息（优先从向量结果中获取，因为包含完整内容）
            doc_info = None
            for result in vector_results:
                if result['id'] == doc_id:
                    doc_info = result
                    break
            
            if not doc_info:
                for result in keyword_results:
                    if result['id'] == doc_id:
                        doc_info = result
                        break
            
            if doc_info:
                rrf_scores.append({
                    'id': doc_id,
                    'title': doc_info.get('title', '无标题'),
                    'content': doc_info.get('content', ''),
                    'author': doc_info.get('author', '未知作者'),
                    'url': doc_info.get('url', ''),
                    'timestamp': doc_info.get('timestamp', ''),
                    'score': total_score,
                    'vector_score': vector_score,
                    'keyword_score': keyword_score,
                    'vector_rank': vector_rank if doc_id in vector_ranks else None,
                    'keyword_rank': keyword_rank if doc_id in keyword_ranks else None
                })
        
        # 按RRF分数降序排序
        rrf_scores.sort(key=lambda x: x['score'], reverse=True)
        
        return rrf_scores[:top_k]
    
    def _create_summary(self, content: str, max_length: int = 100) -> str:
        """
        创建内容摘要
        
        Args:
            content: 原始内容
            max_length: 摘要最大长度
            
        Returns:
            摘要文本
        """
        if not content:
            return ""
        
        # 简单实现：截取前max_length个字符
        if len(content) <= max_length:
            return content
        
        # 尝试在句子边界处截断
        sentences = content.split('。')
        summary = ""
        for sentence in sentences:
            if len(summary) + len(sentence) + 1 <= max_length:
                if summary:
                    summary += "。" + sentence
                else:
                    summary = sentence
            else:
                break
        
        if summary:
            return summary + "。"
        else:
            # 如果无法按句子截断，直接截取
            return content[:max_length] + "..."
    
    def search(self, query: str, top_k: int = 20, fusion_method: str = "rrf") -> List[Dict[str, Any]]:
        """
        执行混合搜索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            fusion_method: 融合方法 (rrf, weighted, simple)
            
        Returns:
            搜索结果列表
        """
        import time
        start_time = time.time()
        
        print(f"执行混合搜索: '{query}' (top_k={top_k}, fusion={fusion_method})")
        
        # 1. 并行执行向量搜索和关键词搜索
        vector_results = self._vector_search(query, top_k=top_k * 2)  # 获取更多结果用于融合
        keyword_results = self._keyword_search(query, top_k=top_k * 2)
        
        print(f"  向量搜索结果: {len(vector_results)} 条")
        print(f"  关键词搜索结果: {len(keyword_results)} 条")
        
        # 2. 结果融合
        if fusion_method == "rrf":
            fused_results = self._rrf_fusion(vector_results, keyword_results, top_k=top_k)
        elif fusion_method == "weighted":
            # 加权融合（简单实现：各占50%）
            fused_results = self._rrf_fusion(vector_results, keyword_results, top_k=top_k)
        else:  # simple
            # 简单合并，优先向量结果
            all_results = {}
            for result in vector_results:
                all_results[result['id']] = result
            
            for result in keyword_results:
                if result['id'] not in all_results:
                    all_results[result['id']] = result
            
            fused_results = list(all_results.values())
            fused_results.sort(key=lambda x: x.get('score', 0), reverse=True)
            fused_results = fused_results[:top_k]
        
        # 3. 添加摘要
        for result in fused_results:
            result['summary'] = self._create_summary(result.get('content', ''))
        
        # 4. 计算搜索时间
        search_time_ms = (time.time() - start_time) * 1000
        print(f"  搜索完成，返回 {len(fused_results)} 条结果，耗时 {search_time_ms:.2f}ms")
        
        return fused_results


# 创建FastAPI应用
app = FastAPI(
    title="校园论坛混合搜索引擎API",
    description="基于向量检索和关键词检索的混合搜索引擎，支持RRF融合",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境中应限制来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局搜索引擎实例
search_engine = None


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化搜索引擎"""
    global search_engine
    try:
        search_engine = HybridSearchEngine()
        print("搜索引擎初始化成功，API服务已就绪")
    except Exception as e:
        print(f"搜索引擎初始化失败: {e}")
        raise


@app.get("/")
async def root():
    """根端点，返回API信息"""
    return {
        "name": "校园论坛混合搜索引擎API",
        "version": "1.0.0",
        "description": "基于向量检索和关键词检索的混合搜索引擎",
        "endpoints": {
            "GET /": "API信息",
            "POST /search": "执行混合搜索",
            "GET /health": "健康检查"
        }
    }


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    执行混合搜索
    
    Args:
        request: 搜索请求，包含查询文本和参数
        
    Returns:
        搜索结果
    """
    if not search_engine:
        raise HTTPException(status_code=503, detail="搜索引擎未初始化")
    
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="查询文本不能为空")
    
    import time
    start_time = time.time()
    
    try:
        # 执行搜索
        results = search_engine.search(
            query=request.query,
            top_k=request.top_k,
            fusion_method=request.fusion_method
        )
        
        # 转换为响应模型
        search_results = []
        for result in results:
            search_results.append(SearchResult(
                id=result['id'],
                title=result['title'],
                content=result['content'],
                author=result['author'],
                url=result['url'],
                timestamp=result['timestamp'],
                score=result['score'],
                summary=result.get('summary', '')
            ))
        
        search_time_ms = (time.time() - start_time) * 1000
        
        return SearchResponse(
            query=request.query,
            total_results=len(search_results),
            results=search_results,
            search_time_ms=search_time_ms
        )
        
    except Exception as e:
        print(f"搜索过程中发生错误: {e}")
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")


@app.get("/health")
async def health_check():
    """健康检查端点"""
    if search_engine:
        return {
            "status": "healthy",
            "engine_initialized": True,
            "chromadb_connected": search_engine.collection is not None,
            "bm25_loaded": search_engine.bm25_model is not None
        }
    else:
        return {
            "status": "unhealthy",
            "engine_initialized": False,
            "message": "搜索引擎未初始化"
        }


@app.get("/stats")
async def get_stats():
    """获取搜索引擎统计信息"""
    if not search_engine:
        raise HTTPException(status_code=503, detail="搜索引擎未初始化")
    
    try:
        chroma_count = search_engine.collection.count() if search_engine.collection else 0
        bm25_count = len(search_engine.bm25_doc_mapping) if search_engine.bm25_doc_mapping else 0
        
        return {
            "chromadb_document_count": chroma_count,
            "bm25_document_count": bm25_count,
            "embedding_model": search_engine.embedding_model_name,
            "status": "operational"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")


def main():
    """主函数，启动FastAPI服务器"""
    import uvicorn
    
    print("=" * 50)
    print("校园论坛混合搜索引擎后端服务")
    print("=" * 50)
    print(f"API文档: http://localhost:8000/docs")
    print(f"健康检查: http://localhost:8000/health")
    print("=" * 50)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False  # 生产环境设为False
    )


if __name__ == "__main__":
    main()
