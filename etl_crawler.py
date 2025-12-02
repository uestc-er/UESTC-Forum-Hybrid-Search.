#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import requests
import json
import time
import os
import re
from typing import List, Dict

class ForumCrawlerFinal:
    def __init__(self, forum_id: int, cookie: str, auth_token: str, max_pages: int = 10):
        self.forum_id = forum_id
        self.max_pages = max_pages
        
        # 1. 列表 API (用于获取帖子清单)
        self.list_api_url = "https://bbs.uestc.edu.cn/_/thread/list"
        
        # 2. 详情 API (根据你提供的准确 URL 修改)
        self.detail_api_url = "https://bbs.uestc.edu.cn/_/post/list"
        
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
            'Referer': f'https://bbs.uestc.edu.cn/forum/{forum_id}',
            'Accept': 'application/json, text/plain, */*',
            'Cookie': cookie,
            'Authorization': auth_token 
        }
        self.session.headers.update(self.headers)

    def fetch_post_list(self, page: int) -> List[Dict]:
        """获取帖子列表"""
        params = {
            'forum_id': self.forum_id,
            'page': page,
            'sort_by': 1,
            'forum_details': 1
        }
        try:
            resp = self.session.get(self.list_api_url, params=params, timeout=10)
            if resp.status_code == 401:
                print("❌ 列表 API 401 未授权！请更新 Token。")
                return []
            resp.raise_for_status()
            data = resp.json()
            # 兼容不同的返回结构
            rows = data.get('data', {}).get('rows', [])
            if not rows:
                rows = data.get('rows', [])
            return rows
        except Exception as e:
            print(f"❌ 获取列表失败: {e}")
            return []

    def fetch_post_detail(self, thread_id: int) -> str:
        """
        【关键修改】使用 /_/post/list 获取详情全文
        """
        params = {
            'thread_id': thread_id,
            'page': 1,
            'thread_details': 1,
            'forum_details': 1
        }
        
        try:
            resp = self.session.get(self.detail_api_url, params=params, timeout=10)
            
            if resp.status_code != 200:
                print(f"    ⚠️ 获取详情失败 HTTP {resp.status_code}")
                return ""
            
            data = resp.json()
            
            # 解析 rows
            # 结构可能是 data['rows'] 或 data['data']['rows']，根据你提供的 JSON 是直接在 data 下？
            # 或者是 data -> rows。通常 API 返回是 {"code":0, "data": { "rows": [...] } }
            # 为了保险，我们做双重检查
            
            rows = []
            if 'rows' in data: 
                rows = data['rows'] # 针对你提供的片段结构
            elif 'data' in data and isinstance(data['data'], dict):
                rows = data['data'].get('rows', [])
            
            if not rows:
                return ""

            # 寻找楼主 (is_first=1)
            target_message = ""
            for row in rows:
                if row.get('is_first') == 1:
                    target_message = row.get('message', '')
                    break
            
            # 如果没找到 is_first，就取第一条
            if not target_message and rows:
                target_message = rows[0].get('message', '')

            return self.clean_text(target_message)

        except Exception as e:
            print(f"    ⚠️ 详情解析异常: {e}")
            return ""

    def clean_text(self, raw_text):
        """清洗文本，处理你提供的示例中的格式"""
        if not raw_text: return ""
        
        # 1. 处理图片/表情代码，例如 ![1155](s)
        # 我们可以把它替换为空，或者替换为 [表情]
        text = re.sub(r'!\[.*?\]\(.*?\)', '', raw_text)
        
        # 2. 去除 HTML 标签 (如果有)
        text = re.sub(r'<[^>]+>', '', text)
        
        # 3. 处理转义字符
        text = text.replace('\n', ' ').replace('\r', '')
        
        return text.strip()

    def crawl(self):
        all_data = []
        print(f"🚀 开始全量爬取 | Forum ID: {self.forum_id}")
        print("💡 提示：爬取全文速度较慢，请耐心等待...")

        for page in range(1, self.max_pages + 1):
            rows = self.fetch_post_list(page)
            if not rows:
                print("⚠️ 本页无数据或已结束。")
                break
            
            print(f"✅ 第 {page} 页: 发现 {len(rows)} 条帖子")
            
            for row in rows:
                thread_id = row.get('thread_id')
                title = row.get('subject')
                author = row.get('author')
                
                # 1. 列表页自带的摘要 (作为备选)
                summary = row.get('summary', '')
                
                # 2. 获取全文
                # 稍微延时，避免并发过高被封
                time.sleep(0.5) 
                full_content = self.fetch_post_detail(thread_id)
                
                # 如果详情页没取到，就用摘要顶替
                final_content = full_content if len(full_content) > len(summary) else summary

                # 时间处理
                try:
                    ts = time.strftime('%Y-%m-%d %H:%M', time.localtime(row.get('dateline', 0)))
                except:
                    ts = "未知时间"

                item = {
                    "id": str(thread_id),
                    "title": title,
                    "author": author,
                    "timestamp": ts,
                    "url": f"https://bbs.uestc.edu.cn/forum.php?mod=viewthread&tid={thread_id}",
                    "content": final_content
                }
                all_data.append(item)
                print(f"  -> {title[:15]}... (正文:{len(final_content)}字)")
            
            self.save_data(all_data)
            
        print(f"\n🎉 爬取结束！共收集 {len(all_data)} 条数据。")

    def save_data(self, data):
        output_file = "data/posts_data.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    # ================= 配置区 =================
    FORUM_ID = 370
    
    # ⚠️ 请务必更新 Cookie 和 Token，因为它们有效期很短
    COOKIE = ""
    AUTH_TOKEN = ""
    # =========================================
    
    if "你的" in AUTH_TOKEN:
        print("❌ 请填入 Cookie 和 Token")
        return

    crawler = ForumCrawlerFinal(FORUM_ID, COOKIE, AUTH_TOKEN, max_pages=5)
    crawler.crawl()

if __name__ == "__main__":
    main()