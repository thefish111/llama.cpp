#!/bin/bash
rm -rf build
cmake -B build -DGGML_CANN=on -DCMAKE_BUILD_TYPE=debug
cmake --build build --config debug -j
# ./build/bin/test-backend-ops test -b CANN0 -o MUL_MAT 

# 运行针对 Q4_1 和 Q8_1 的测试
echo "运行 test-backend-ops 测试..."
echo "过滤包含 Q4_1 和 Q8_1 的 MUL_MAT 测试..."
echo ""

# 如果 build 目录存在且有可执行文件
if [ -f "./build/bin/test-backend-ops" ]; then
    # 运行测试并过滤 Q4_1 和 Q8_1 相关输出
    ./build/bin/test-backend-ops test -b CANN0 -o MUL_MAT 2>&1 | grep -E "(Q4_1|Q8_1|MUL_MAT)" 
else
    echo "❌ 错误: ./build/bin/test-backend-ops 不存在"
    echo "请先编译项目："
    echo "  cmake -B build -DGGML_CANN=ON -DCMAKE_BUILD_TYPE=Release"
    echo "  cmake --build build --config Release -j $(nproc)"
fi
