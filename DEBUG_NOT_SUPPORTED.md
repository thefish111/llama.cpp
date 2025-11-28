# 调试 Q4_1/Q8_1 "not supported [CANN0]" 问题

## 问题定位

你的 "not supported [CANN0]" 表示 `ggml_backend_cann_supports_op()` 返回了 false。

**函数位置**：`ggml/src/ggml-cann/ggml-cann.cpp` 第 2521-2571 行

## 返回 FALSE 的 5 种情况

### ❌ 情况 1：类型不支持
```cpp
default:
    return false;  // src[0] 类型不在支持列表中
```
**你的状态**：✅ 已修复（已添加 Q4_1 和 Q8_1）

---

### ❌ 情况 2：ASCEND_310P 限制
```cpp
#ifdef ASCEND_310P
    return false;  // 310P 硬件不支持 Q4/Q8
#endif
```
**检查方法**：
```bash
cat build/CMakeCache.txt | grep SOC_TYPE
```
**输出示例**：
- 如果包含 `310P`：说明在 310P 上，Q4/Q8 不支持（硬件限制）
- 如果是 `910B` 或其他：不是这个问题

---

### ❌ 情况 3：src[0] 不 contiguous（权重）
```cpp
return ggml_is_contiguous(op->src[0]) && ...  // 权重不连续
```
**导致的原因**：
- 测试用例使用了 `permute`（维度重排）
- 测试用例使用了 `view`（视图）
- 测试用例使用了 `transpose`（转置）

**示例测试用例**：
```cpp
// 这些会导致 src[0] 不连续：
test_mul_mat(..., per={0, 2, 1, 3})  // permutation
test_mul_mat(..., v=true)            // view
```

---

### ❌ 情况 4：src[1] 不 contiguous（输入）
```cpp
return ... && ggml_is_contiguous(op->src[1])  // 输入不连续
```
**导致的原因**：同上

---

### ❌ 情况 5：不是 MUL_MAT 操作
**你的状态**：✅ 不可能（你在测试 MUL_MAT）

---

## 🔍 使用调试版本定位问题

### 步骤 1：启用调试输出

已在代码中添加调试功能（受环境变量控制），重新编译后运行：

```bash
# 重新编译
cmake --build build --config Release -j $(nproc)

# 启用调试输出并运行测试
export CANN_DEBUG_SUPPORTS_OP=1
./build/bin/test-backend-ops test -b CANN0 -o MUL_MAT 2>&1 | grep -E "(Q4_1|Q8_1|CANN)"
```

### 步骤 2：解读输出

#### 输出示例 A：contiguous 失败
```
[CANN] MUL_MAT type=Q4_1 src0_contig=0 src1_contig=1 => NOT_SUPPORTED
```
**解释**：
- `src0_contig=0`：权重不连续 ❌
- `src1_contig=1`：输入连续 ✅
- **结论**：测试用例使用了 permutation/view，导致权重不连续

**这是正常的**！CANN 的量化矩阵乘法要求 tensor 连续。

---

#### 输出示例 B：ASCEND_310P
```
[CANN] MUL_MAT type=Q4_1: REJECTED (ASCEND_310P)
```
**解释**：在 310P 硬件上，不支持 Q4/Q8

**这是硬件限制**！只能在 910 系列上使用。

---

#### 输出示例 C：都连续，应该支持
```
[CANN] MUL_MAT type=Q4_1 src0_contig=1 src1_contig=1 => SUPPORTED
```
**解释**：supports_op 返回 true，测试应该运行

**如果仍然显示 "not supported"**：
- 检查是否有其他 tensor 在图中（测试框架会检查所有 tensor）
- 检查参数 tensor 是否是 F32（梯度要求）

---

#### 输出示例 D：类型不支持
```
[CANN] MUL_MAT type=Q4_K: UNSUPPORTED_TYPE
```
**解释**：其他量化类型（如 Q4_K）不支持

**这是正常的**！只支持 Q4_0/Q4_1/Q8_0/Q8_1。

---

## 📊 预期的测试结果

### ✅ 正常的 "not supported"（不是错误）

```
# 带 permutation 的测试
test_mul_mat: type_a=Q4_1, type_b=F32, ... per={0,2,1,3} ... not supported [CANN0]
[CANN] MUL_MAT type=Q4_1 src0_contig=0 src1_contig=1 => NOT_SUPPORTED

# 带 view 的测试
test_mul_mat: type_a=Q4_1, type_b=F32, ... v=1 ... not supported [CANN0]
[CANN] MUL_MAT type=Q4_1 src0_contig=0 src1_contig=1 => NOT_SUPPORTED

# 参数类型检查
test_mul_mat: type_a=Q4_1, type_b=F32, ... not supported [a->type != FP32]
```

**这些都是正常行为**！因为：
1. CANN 量化操作要求 contiguous
2. 量化权重不能作为可训练参数（需要 F32）

---

### ✅ 成功的测试（这才是重点！）

```
# 简单测试用例
test_mul_mat: type_a=Q4_1, type_b=F32, m=16, n=16, k=256, bs={1,1}, nr={1,1} ... OK
[CANN] MUL_MAT type=Q4_1 src0_contig=1 src1_contig=1 => SUPPORTED

test_mul_mat: type_a=Q8_1, type_b=F32, m=16, n=16, k=256, bs={1,1}, nr={1,1} ... OK
[CANN] MUL_MAT type=Q8_1 src0_contig=1 src1_contig=1 => SUPPORTED
```

**看到 OK 就说明成功了**！✅

---

### ❌ 失败的测试（需要调试）

```
# 支持但计算错误
test_mul_mat: type_a=Q4_1, type_b=F32, m=16, n=16, k=256, bs={1,1}, nr={1,1} ... FAIL
[CANN] MUL_MAT type=Q4_1 src0_contig=1 src1_contig=1 => SUPPORTED
```

**如果看到 FAIL**：说明：
1. ✅ supports_op 正确返回 true
2. ❌ 但计算结果错误

**需要检查**：
- transform 函数是否正确
- offset tensor 创建是否正确
- 算子调用参数是否正确

---

## 🎯 快速验证步骤

### 1. 检查是否 310P
```bash
cat build/CMakeCache.txt | grep SOC_TYPE
# 或
npu-smi info | grep -i "chip name"
```

### 2. 运行简单测试
```bash
export CANN_DEBUG_SUPPORTS_OP=1
./build/bin/test-backend-ops test -b CANN0 -o MUL_MAT 2>&1 | \
  grep -A1 "type_a=Q4_1.*bs={1, 1}.*nr={1, 1}"
```

### 3. 查看关键输出
重点关注：
- `src0_contig=1 src1_contig=1 => SUPPORTED` ✅
- 后面跟着 `OK` 或 `FAIL`

### 4. 统计结果
```bash
# 统计 Q4_1 支持情况
./build/bin/test-backend-ops test -b CANN0 -o MUL_MAT 2>&1 | \
  grep "Q4_1" | grep -c "OK"

# 统计 Q8_1 支持情况
./build/bin/test-backend-ops test -b CANN0 -o MUL_MAT 2>&1 | \
  grep "Q8_1" | grep -c "OK"
```

**如果数字 > 0**：说明有测试通过了！✅

---

## 📋 总结

**"not supported [CANN0]" 的原因优先级**：

1. ⚠️ **最可能**：contiguous 检查失败（测试用例使用了 permutation/view）
   - **这是正常的**！不是错误

2. ❓ **可能**：ASCEND_310P 硬件限制
   - 检查 SOC_TYPE

3. ✅ **不太可能**：类型不支持
   - 你已经添加了 Q4_1/Q8_1

**重点关注**：
- 寻找 `src0_contig=1 src1_contig=1 => SUPPORTED` 后面跟 `OK` 的测试
- 忽略带 permutation/view 的 "not supported"
- 如果看到 `FAIL`，才需要调试实现代码

