#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混合搜索引擎前端界面 (Streamlit)
功能：提供Web交互界面，调用后端API进行混合检索
"""

import streamlit as st
import requests
import json
from datetime import datetime
from typing import List, Dict, Any
import time


# 配置
BACKEND_URL = "http://localhost:8000"
SEARCH_ENDPOINT = f"{BACKEND_URL}/search"
HEALTH_ENDPOINT = f"{BACKEND_URL}/health"
STATS_ENDPOINT = f"{BACKEND_URL}/stats"

# 页面配置
st.set_page_config(
    page_title="校园论坛混合搜索引擎",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        background-color: #f9f9f9;
        transition: box-shadow 0.3s;
    }
    .result-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .result-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .result-meta {
        font-size: 0.9rem;
        color: #666;
        margin-bottom: 0.5rem;
    }
    .result-summary {
        font-size: 1rem;
        color: #333;
        line-height: 1.5;
    }
    .score-badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 12px;
        background-color: #E3F2FD;
        color: #1E88E5;
        font-size: 0.8rem;
        font-weight: bold;
        margin-right: 0.5rem;
    }
    .search-time {
        font-size: 0.9rem;
        color: #4CAF50;
        font-style: italic;
    }
    .stButton button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def check_backend_health() -> bool:
    """检查后端服务是否健康"""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("status") == "healthy"
        return False
    except:
        return False


def get_backend_stats() -> Dict[str, Any]:
    """获取后端统计信息"""
    try:
        response = requests.get(STATS_ENDPOINT, timeout=5)
        if response.status_code == 200:
            return response.json()
        return {}
    except:
        return {}


def perform_search(query: str, top_k: int = 20, fusion_method: str = "rrf") -> Dict[str, Any]:
    """
    执行搜索
    
    Args:
        query: 搜索查询
        top_k: 返回结果数量
        fusion_method: 融合方法
        
    Returns:
        搜索结果
    """
    try:
        payload = {
            "query": query,
            "top_k": top_k,
            "fusion_method": fusion_method
        }
        
        response = requests.post(SEARCH_ENDPOINT, json=payload, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"搜索失败: {response.status_code} - {response.text}")
            return {"query": query, "total_results": 0, "results": [], "search_time_ms": 0}
            
    except requests.exceptions.ConnectionError:
        st.error("无法连接到后端服务，请确保 server.py 正在运行")
        return {"query": query, "total_results": 0, "results": [], "search_time_ms": 0}
    except Exception as e:
        st.error(f"搜索过程中发生错误: {str(e)}")
        return {"query": query, "total_results": 0, "results": [], "search_time_ms": 0}


def format_timestamp(timestamp_str: str) -> str:
    """格式化时间戳"""
    if not timestamp_str:
        return "未知时间"
    
    try:
        # 尝试解析时间戳（可能是Unix时间戳或字符串）
        if timestamp_str.isdigit():
            # Unix时间戳
            dt = datetime.fromtimestamp(int(timestamp_str))
        else:
            # 字符串时间
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        
        return dt.strftime("%Y-%m-%d %H:%M")
    except:
        return timestamp_str


def display_result(result: Dict[str, Any], index: int):
    """显示单个搜索结果"""
    with st.container():
        st.markdown(f"""
        <div class="result-card">
            <div class="result-title">
                {index}. <a href="{result['url']}" target="_blank">{result['title']}</a>
            </div>
            <div class="result-meta">
                <span class="score-badge">相关度: {result['score']:.3f}</span>
                <span>作者: {result['author']}</span>
                <span> | </span>
                <span>发布时间: {format_timestamp(result['timestamp'])}</span>
            </div>
            <div class="result-summary">
                {result['summary']}
            </div>
        </div>
        """, unsafe_allow_html=True)


def main():
    """主函数"""
    # 页眉
    st.markdown('<div class="main-header">🔍 校园论坛混合搜索引擎</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">语义 + 关键词混合检索 · 倒数排名融合(RRF) · 智能摘要</div>', unsafe_allow_html=True)
    
    # 侧边栏
    with st.sidebar:
        st.header("⚙️ 搜索设置")
        
        # 检查后端健康状态
        backend_healthy = check_backend_health()
        
        if backend_healthy:
            st.success("✅ 后端服务连接正常")
            
            # 显示统计信息
            stats = get_backend_stats()
            if stats:
                st.info("📊 索引统计")
                st.write(f"向量索引文档数: {stats.get('chromadb_document_count', 0)}")
                st.write(f"关键词索引文档数: {stats.get('bm25_document_count', 0)}")
                st.write(f"嵌入模型: {stats.get('embedding_model', '未知')}")
        else:
            st.error("❌ 后端服务未连接")
            st.warning("请先启动后端服务:")
            st.code("python server.py")
            st.info("后端启动后，将运行在: http://localhost:8000")
        
        st.markdown("---")
        
        # 搜索参数设置
        top_k = st.slider(
            "返回结果数量",
            min_value=5,
            max_value=50,
            value=20,
            help="每次搜索返回的结果数量"
        )
        
        fusion_method = st.selectbox(
            "融合方法",
            options=["rrf", "weighted", "simple"],
            index=0,
            help="RRF: 倒数排名融合（推荐）\n加权: 加权融合\n简单: 简单合并"
        )
        
        st.markdown("---")
        
        # 技术说明
        st.header("ℹ️ 技术说明")
        st.markdown("""
        **混合检索流程:**
        1. **向量检索**: 使用Sentence Transformers计算语义相似度
        2. **关键词检索**: 使用BM25算法进行关键词匹配
        3. **RRF融合**: 使用倒数排名融合算法合并两种结果
        4. **智能摘要**: 自动生成内容摘要
        
        **核心特性:**
        - 支持语义理解和关键词匹配
        - 自适应结果融合
        - 实时搜索响应
        - 友好的用户界面
        """)
        
        st.markdown("---")
        
        # 使用说明
        st.header("📖 使用说明")
        st.markdown("""
        1. 确保后端服务已启动
        2. 在搜索框中输入查询
        3. 点击"开始搜索"按钮
        4. 查看混合检索结果
        5. 点击标题可跳转到原帖
        """)
    
    # 主内容区
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # 搜索框
        query = st.text_input(
            "",
            placeholder="请输入搜索内容，例如：研究生选课建议、校园活动推荐、宿舍问题咨询...",
            key="search_input"
        )
    
    with col2:
        st.write("")  # 垂直间距
        st.write("")  # 垂直间距
        search_button = st.button("🔍 开始搜索", type="primary", use_container_width=True)
    
    # 搜索历史（会话状态）
    if "search_history" not in st.session_state:
        st.session_state.search_history = []
    
    # 执行搜索
    if search_button and query:
        if not backend_healthy:
            st.error("后端服务未连接，无法执行搜索")
            st.info("请先启动后端服务: `python server.py`")
        else:
            with st.spinner("正在执行混合检索，请稍候..."):
                # 执行搜索
                search_results = perform_search(query, top_k=top_k, fusion_method=fusion_method)
                
                # 保存到搜索历史
                if query not in st.session_state.search_history:
                    st.session_state.search_history.insert(0, query)
                    # 只保留最近10条历史
                    if len(st.session_state.search_history) > 10:
                        st.session_state.search_history = st.session_state.search_history[:10]
                
                # 显示搜索结果
                st.markdown("---")
                
                if search_results["total_results"] > 0:
                    # 显示搜索统计
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("搜索查询", search_results["query"])
                    with col_b:
                        st.metric("找到结果", search_results["total_results"])
                    with col_c:
                        st.metric("搜索耗时", f"{search_results['search_time_ms']:.0f}ms")
                    
                    st.markdown("---")
                    
                    # 显示结果
                    st.subheader("📄 搜索结果")
                    
                    for i, result in enumerate(search_results["results"], 1):
                        display_result(result, i)
                    
                    # 显示技术细节（可折叠）
                    with st.expander("🔧 查看技术细节"):
                        st.json(search_results)
                        
                else:
                    st.warning("未找到相关结果，请尝试其他搜索词")
    
    # 如果没有执行搜索，显示搜索历史或提示
    elif not query and st.session_state.search_history:
        st.markdown("---")
        st.subheader("📜 搜索历史")
        
        for i, history_query in enumerate(st.session_state.search_history[:5]):
            if st.button(f"{i+1}. {history_query}", key=f"history_{i}"):
                # 当点击历史记录时，填充搜索框并执行搜索
                st.session_state.search_input = history_query
                st.rerun()
    
    # 页脚
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>校园论坛混合搜索引擎 v1.0 | 基于 FastAPI + Streamlit + ChromaDB + BM25</p>
        <p>技术支持: 向量检索 · 关键词检索 · RRF融合 · 智能摘要</p>
    </div>
    """, unsafe_allow_html=True)


def run_app():
    """运行Streamlit应用"""
    try:
        main()
    except Exception as e:
        st.error(f"应用运行错误: {str(e)}")
        st.info("请检查后端服务是否正常运行")


if __name__ == "__main__":
    run_app()
