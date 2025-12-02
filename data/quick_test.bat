@echo off
echo 快速测试混合搜索引擎...
echo.
echo 1. 构建索引...
python build_index.py
echo.
echo 2. 启动后端服务（新窗口）...
start python server.py
timeout /t 3 /nobreak >nul
echo.
echo 3. 测试搜索API...
curl -X POST http://localhost:8000/search ^
  -H "Content-Type: application/json" ^
  -d "{"query": "如何选课", "top_k": 3}"
echo.
echo.
echo 测试完成！按任意键退出...
pause >nul
