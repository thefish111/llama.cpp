// 在 ggml-cann.cpp 的 supports_op 函数中添加调试日志
// 位置：case GGML_OP_MUL_MAT 部分

// 原始代码（第 2521-2540 行）：
/*
case GGML_OP_MUL_MAT: {
    switch (op->src[0]->type) {
        case GGML_TYPE_F16:
        case GGML_TYPE_F32:
            return true;
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q8_1:
#ifdef ASCEND_310P
            // Q4 && Q8 per group is not suppor on 310p device
            return false;
#endif
            // only support contiguous for quantized types.
            return ggml_is_contiguous(op->src[0]) &&
                    ggml_is_contiguous(op->src[1]);
        default:
            return false;
    }
}
*/

// ========== 调试版本：替换上面的代码 ==========
case GGML_OP_MUL_MAT: {
    // 添加调试信息（可以通过环境变量控制）
    static bool debug_mode = getenv("CANN_DEBUG_SUPPORTS_OP") != nullptr;
    
    if (debug_mode) {
        fprintf(stderr, "[CANN DEBUG] MUL_MAT supports_op check:\n");
        fprintf(stderr, "  - src[0] type: %s (name: %s)\n", 
                ggml_type_name(op->src[0]->type), op->src[0]->name);
        fprintf(stderr, "  - src[1] type: %s (name: %s)\n", 
                ggml_type_name(op->src[1]->type), op->src[1]->name);
        fprintf(stderr, "  - src[0] contiguous: %s\n", 
                ggml_is_contiguous(op->src[0]) ? "true" : "false");
        fprintf(stderr, "  - src[1] contiguous: %s\n", 
                ggml_is_contiguous(op->src[1]) ? "true" : "false");
#ifdef ASCEND_310P
        fprintf(stderr, "  - ASCEND_310P: DEFINED\n");
#else
        fprintf(stderr, "  - ASCEND_310P: not defined\n");
#endif
    }
    
    switch (op->src[0]->type) {
        case GGML_TYPE_F16:
        case GGML_TYPE_F32:
            if (debug_mode) fprintf(stderr, "  => Returning true (FP16/FP32)\n");
            return true;
            
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q8_1:
#ifdef ASCEND_310P
            // Q4 && Q8 per group is not suppor on 310p device
            if (debug_mode) fprintf(stderr, "  => Returning false (ASCEND_310P)\n");
            return false;
#endif
            {
                // only support contiguous for quantized types.
                bool is_src0_contig = ggml_is_contiguous(op->src[0]);
                bool is_src1_contig = ggml_is_contiguous(op->src[1]);
                bool result = is_src0_contig && is_src1_contig;
                
                if (debug_mode) {
                    fprintf(stderr, "  => Contiguous check: src0=%s, src1=%s, result=%s\n",
                            is_src0_contig ? "Y" : "N",
                            is_src1_contig ? "Y" : "N",
                            result ? "true (SUPPORTED)" : "false (NOT SUPPORTED)");
                }
                
                return result;
            }
            
        default:
            if (debug_mode) {
                fprintf(stderr, "  => Returning false (unsupported type: %s)\n", 
                        ggml_type_name(op->src[0]->type));
            }
            return false;
    }
}

// ========== 使用方法 ==========
// 1. 替换 ggml-cann.cpp 中的 MUL_MAT 支持检查代码
// 2. 重新编译
// 3. 运行测试时设置环境变量：
//    export CANN_DEBUG_SUPPORTS_OP=1
//    ./build/bin/test-backend-ops test -b CANN0 -o MUL_MAT
//
// 输出示例：
// [CANN DEBUG] MUL_MAT supports_op check:
//   - src[0] type: Q4_1 (name: a)
//   - src[1] type: F32 (name: b)
//   - src[0] contiguous: true
//   - src[1] contiguous: true
//   - ASCEND_310P: not defined
//   => Contiguous check: src0=Y, src1=Y, result=true (SUPPORTED)
//
// 如果看到 result=false，就知道具体是哪个检查失败了！

// ========== 可能的输出和原因 ==========
//
// 情况 1：src[0] contiguous: false
// 原因：权重 tensor 不连续（可能使用了 view、permute 等操作）
// 解决：检查 test_mul_mat 中的 permutation 测试用例
//
// 情况 2：src[1] contiguous: false  
// 原因：输入 tensor 不连续
// 解决：检查测试用例的 view 参数 (v=true)
//
// 情况 3：ASCEND_310P: DEFINED
// 原因：在 310P 硬件上，不支持 Q4/Q8
// 解决：这是硬件限制，无法通过代码修复
//
// 情况 4：unsupported type: XXX
// 原因：src[0] 的类型不在支持列表中
// 解决：检查是否是新的量化类型（如 Q4_K 等）

