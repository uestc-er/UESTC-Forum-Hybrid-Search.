#!/bin/bash
echo "快速测试混合搜索引擎..."
echo "1. 构建索引..."
python build_index.py
echo ""
echo "2. 启动后端服务（后台运行）..."
python server.py &
SERVER_PID=$!
sleep 3
echo ""
echo "3. 测试搜索API..."
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "如何选课", "top_k": 3}'
echo ""
echo ""
echo "4. 停止后端服务..."
kill $SERVER_PID
echo "测试完成！"
