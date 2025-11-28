# 诊断 "not supported" 的原因

## 如何识别具体原因

当测试输出 `not supported [reason]` 时，通过 `[reason]` 的内容可以判断：

### 情况 1：`not supported [CANN0]` 或 `not supported [backend_name]`
**原因**：`ggml_backend_supports_op()` 返回 false

**可能的子原因**：
1. **操作类型不支持**：后端根本不支持这个操作（如某些特殊 OP）
2. **数据类型组合不支持**：
   - `op->src[0]->type` 不在支持列表中
   - 对于 MUL_MAT，检查是否通过了类型判断
3. **Tensor 不是 contiguous**：
   ```cpp
   return ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]);
   ```
4. **特定硬件限制**（如 ASCEND_310P）

**调试方法**：
- 在 `ggml-cann.cpp` 的 `supports_op` 函数中添加日志
- 检查 `op->src[0]->type` 和 `op->src[1]->type`
- 检查 `ggml_is_contiguous()` 的返回值

### 情况 2：`not supported [tensor_name->type != FP32]`
**原因**：参数 tensor（带 GGML_TENSOR_FLAG_PARAM 标志）不是 F32 类型

**示例**：`not supported [a->type != FP32]` 或 `not supported [b->type != FP32]`

**解释**：
- 梯度计算要求参数必须是 F32
- 量化类型的 tensor 不能作为可训练参数
- 这是**正常行为**！量化矩阵乘法中，权重（type_a）是量化的但**不应该**标记为参数

**这种情况是正常的**：对于量化类型，权重不参与梯度计算，所以测试会跳过。

### 情况 3：`not supported [operation description]`
**原因**：计算图中没有任何参数 tensor

**解释**：
- 测试框架需要至少一个可训练参数来测试梯度
- 如果所有 tensor 都不是参数，测试无法进行
- 这通常发生在量化推理场景

---

## 针对 Q4_1 和 Q8_1 的诊断步骤

### 步骤 1：确认编译配置

检查是否定义了 ASCEND_310P：
```bash
grep -i "ASCEND_310P" build/CMakeCache.txt
# 如果有输出，说明 310P 模式被启用，Q4/Q8 会被禁用
```

### 步骤 2：添加调试日志

在 `ggml/src/ggml-cann/ggml-cann.cpp` 的 `supports_op` 函数中添加日志：

```cpp
case GGML_OP_MUL_MAT: {
    printf("[DEBUG] MUL_MAT supports_op check:\n");
    printf("  - op->src[0]->type = %s\n", ggml_type_name(op->src[0]->type));
    printf("  - op->src[1]->type = %s\n", ggml_type_name(op->src[1]->type));
    printf("  - src[0] contiguous = %d\n", ggml_is_contiguous(op->src[0]));
    printf("  - src[1] contiguous = %d\n", ggml_is_contiguous(op->src[1]));
    
    switch (op->src[0]->type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q8_1:
#ifdef ASCEND_310P
            printf("  - ASCEND_310P is defined, returning false\n");
            return false;
#endif
            bool result = ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]);
            printf("  - Final result = %d\n", result);
            return result;
        default:
            return false;
    }
}
```

### 步骤 3：查看测试输出模式

运行测试并观察输出：

```bash
./build/bin/test-backend-ops test -b CANN0 -o MUL_MAT 2>&1 | grep -A 2 "Q4_1\|Q8_1"
```

**解读输出**：

1. **如果看到很多 `not supported [CANN0]`**：
   - 说明 `supports_op` 返回 false
   - 可能是 contiguous 检查失败
   - 可能是 ASCEND_310P 被定义

2. **如果看到 `not supported [a->type != FP32]`**：
   - **这是正常的**！
   - 量化权重不作为参数，测试跳过梯度检查
   - 这不是错误

3. **如果看到 `OK` 或 `FAIL`**：
   - `OK`：测试通过！✅
   - `FAIL`：支持声称但计算错误 ❌

### 步骤 4：针对性测试

创建简单的测试用例：

```cpp
// 在 test-backend-ops.cpp 中添加
test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q4_1, GGML_TYPE_F32, 32, 32, 256, {1, 1}, {1, 1}));
test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q8_1, GGML_TYPE_F32, 32, 32, 256, {1, 1}, {1, 1}));
```

单独运行这些测试查看具体输出。

---

## 常见的 "not supported" 场景总结

| 输出信息 | 原因 | 是否正常 | 解决方法 |
|---------|------|---------|---------|
| `not supported [CANN0]` | `supports_op` 返回 false | 取决于具体情况 | 检查类型支持和 contiguous |
| `not supported [a->type != FP32]` | 量化参数不是 F32 | ✅ 正常 | 无需修复 |
| `not supported [b->type != FP32]` | 输入参数不是 F32 | ✅ 正常 | 无需修复 |
| `not supported [MUL_MAT(...)]` | 没有参数 tensor | ✅ 正常 | 无需修复 |

---

## 你当前的情况分析

由于你在 macOS 上：
1. **无法编译 CANN 后端**（CMake 会失败）
2. **无法运行实际测试**
3. **代码修改已完成**，但需要在 Linux + Ascend NPU 上验证

**建议**：
- 将代码提交到有 Ascend 硬件的机器上测试
- 或者请求能访问该硬件的同事帮忙测试
- 重点关注是否有 `OK` 或 `FAIL` 输出，而不是 `not supported`
