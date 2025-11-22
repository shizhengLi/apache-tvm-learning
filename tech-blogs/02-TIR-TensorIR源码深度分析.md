# TIR (Tensor IR) 源码深度分析

## 1. TIR 概述

Tensor IR (TIR) 是TVM的核心中间表示，用于在张量级别表示计算。它是一种低级的循环嵌套表示，包含了缓冲区、循环、计算块等概念，适合描述单个算子的具体实现细节。

### 1.1 TIR的核心特点

1. **显式内存管理**：通过Buffer抽象提供精确的内存布局控制
2. **循环嵌套表示**：可以直接操作循环结构和计算块
3. **调度友好**：设计为易于进行各种程序变换和优化
4. **设备无关**：可以在不同设备上进行代码生成

### 1.2 TIR在TVM架构中的位置

```
前端模型 → Relax IR → TIR (Tensor IR) → 目标代码
                         ↑
                    调度变换
```

## 2. TIR 核心数据结构

### 2.1 PrimFunc - TIR函数的基础

PrimFunc是TIR的基本函数单位，位于 `include/tvm/tir/function.h`：

```cpp
class PrimFuncNode : public BaseFuncNode {
 public:
  /*! \brief Function parameters */
  ffi::Array<tir::Var> params;
  /*! \brief The return type of the function. */
  Type ret_type;
  /*! \brief Maps some parameters to specific Buffer data structures. */
  ffi::Map<tir::Var, Buffer> buffer_map;
  /*! \brief The body of the function */
  Stmt body;
  // ... 其他字段
};
```

**关键设计点分析：**

1. **buffer_map机制**：
   - 提供参数到Buffer的映射
   - 支持形状约束检查
   - 简化参数解包和约束验证

2. **类型系统**：
   - 支持静态类型检查
   - 返回类型可以明确指定

**Python接口示例：**
```python
from tvm import tir

@tir.prim_func
def matmul_func(a: tir.handle, b: tir.handle, c: tir.handle) -> None:
    A = tir.match_buffer(a, (1024, 1024), "float32")
    B = tir.match_buffer(b, (1024, 1024), "float32")
    C = tir.match_buffer(c, (1024, 1024), "float32")

    with tir.block([1024, 1024], "C") as [vi, vj]:
        tir.bind(C[vi, vj], 0.0)
        with tir.block([1024], "update") as [vk]:
            C[vi, vj] += A[vi, vk] * B[vk, vj]
```

### 2.2 表达式系统 (Expressions)

TIR的表达式系统继承自基础的PrimExpr，位于 `include/tvm/tir/expr.h`：

```cpp
// 基础表达式类型
class IntImmNode : public PrimExprNode
class FloatImmNode : public PrimExprNode
class StringImmNode : public PrimExprNode

// 变量
class VarNode : public PrimExprNode
class SizeVarNode : public VarNode

// 运算表达式
class AddNode : public PrimExprNode
class SubNode : public PrimExprNode
class MulNode : public PrimExprNode
class DivNode : public PrimExprNode

// 特殊表达式
class CastNode : public PrimExprNode
class RampNode : public PrimExprNode
class BroadcastNode : public PrimExprNode

// 内存访问表达式
class LoadNode : public PrimExprNode
class BufferLoadNode : public PrimExprNode
```

**表达式层次结构设计：**

1. **继承关系**：所有表达式都继承自PrimExprNode
2. **类型安全**：编译时类型检查
3. **FFI支持**：与Python的无缝集成

### 2.3 语句系统 (Statements)

TIR的语句系统定义了程序的控制流，位于 `include/tvm/tir/stmt.h`：

```cpp
class StmtNode : public Object {
 public:
  mutable Span span;  // 调试信息
  // ...
};

// 基础语句
class LetStmtNode : public StmtNode {
  Var var;
  PrimExpr value;
  Stmt body;
};

class AttrStmtNode : public StmtNode {
  PrimExpr node;
  String attr_key;
  PrimExpr value;
  Stmt body;
};

// 控制流语句
class ForNode : public StmtNode {
  Var loop_var;
  PrimExpr min;
  PrimExpr extent;
  ForKind kind;  // Serial, Parallel, Vectorized, Unrolled
  Stmt body;
};

class IfThenElseNode : public StmtNode {
  PrimExpr condition;
  Stmt then_case;
  Stmt else_case;
};

// 存储语句
class StoreNode : public StmtNode {
  Var buffer_var;
  PrimExpr index;
  PrimExpr value;
  PrimExpr predicate;
};

class BufferStoreNode : public StmtNode {
  Buffer buffer;
  PrimExpr value;
  Array<PrimExpr> indices;
};

// 计算块语句 (TIR特有的高级抽象)
class BlockNode : public StmtNode {
  Array<IterVar> iter_vars;        // 迭代变量
  Array<BufferRegion> reads;       // 读缓冲区
  Array<BufferRegion> writes;      // 写缓冲区
  String name_hint;               // 块名称
  Optional<Stmt> init;            // 初始化语句
  Stmt body;                      // 计算体
  Array<Buffer> alloc_buffers;    // 分配的缓冲区
  Optional<MatchBufferRegion> match_buffers;  // 缓冲区匹配
  Map<String, ObjectRef> annotations;         // 注解
};
```

## 3. TIR 调度系统

### 3.1 Schedule类的核心设计

Schedule类提供了对TIR程序进行结构化变换的接口，位于 `include/tvm/tir/schedule/schedule.h`：

```cpp
class ScheduleNode : public Object {
 public:
  /*! \brief The state of the schedule */
  ScheduleState state;
  /*! \brief The trace of instructions applied */
  Optional<Trace> trace;
  /*! \brief Random state for reproducible random sampling */
  support::RandomGenerator::TRandState rand_state;
};

// 随机变量类型
class BlockRV : public runtime::ObjectRef;  // 代表一个Block
class LoopRV : public runtime::ObjectRef;   // 代表一个Loop
class ExprRV : public runtime::ObjectRef;   // 代表一个表达式
```

### 3.2 调度原语 (Scheduling Primitives)

TVM提供了丰富的调度原语来变换TIR程序：

#### 循环变换
```cpp
// 分块
Array<LoopRV> split(Schedule self, LoopRV loop, Array<PrimExpr> factors);
LoopRV fuse(Schedule self, Array<LoopRV> loops);
LoopRV reorder(Schedule self, Array<LoopRV> loops);

// 循环属性设置
void vectorize(Schedule self, LoopRV loop);
void parallel(Schedule self, LoopRV loop);
void unroll(Schedule self, LoopRV loop);
```

#### 计算块变换
```cpp
// 计算块操作
BlockRV get_block(Schedule self, String name, String func_name);
void compute_inline(Schedule self, BlockRV block);
void reverse_compute_inline(Schedule self, BlockRV block);

// 缓存操作
BlockRV cache_read(Schedule self, BlockRV block, int read_buffer_index,
                   String storage_scope);
BlockRV cache_write(Schedule self, BlockRV block, int write_buffer_index,
                    String storage_scope);

// 计算重排
void compute_at(Schedule self, BlockRV block, LoopRV loop, bool preserve_unit_loops);
void reverse_compute_at(Schedule self, BlockRV block, LoopRV loop);
```

### 3.3 调度状态管理

ScheduleState维护了调度的完整状态信息：

```cpp
class ScheduleStateNode : public Object {
 public:
  /*! \brief The mod being scheduled */
  IRModule mod;
  /*! \brief The function to be scheduled */
  tir::PrimFunc func;
  /*! \brief Symbolic table for random variables */
  Map<ObjectRef, ObjectRef> sref2obj;
  /*! \brief Reverse mapping for quick lookup */
  Map<tir::Stmt, tir::StmtSRef> obj2sref;
  /*! \brief Scope information of each block */
  Map<tir::Block, ScopeInfo> block_scope;
  // ... 更多状态信息
};
```

## 4. TIR 分析与验证

### 4.1 格式验证

TVM提供了完整的TIR程序格式验证，位于 `src/tir/analysis/verify_well_formed.cc`：

```cpp
template <typename DerivedVerifier>
class Verifier : protected TIRVisitorWithPath {
 protected:
  explicit Verifier(bool assert_on_error) : assert_on_error_(assert_on_error) {}

  template <typename TirNodeRef>
  static bool Verify(const TirNodeRef& node, bool assert_on_error) {
    DerivedVerifier verifier(assert_on_error);
    verifier(node);
    return !verifier.has_error_;
  }
};
```

**验证规则包括：**

1. **变量作用域检查**：确保变量在使用前已定义
2. **类型一致性**：表达式类型匹配
3. **内存访问有效性**：数组越界检查
4. **SSA格式**：单一赋值格式验证

### 4.2 依赖分析

TVM提供多种依赖分析来支持调度优化：

```cpp
// 块依赖分析
class BlockDependencyAnalyzer {
 public:
  // 检查两个块之间是否存在依赖
  bool HasDependency(Block src, Block dst);

  // 计算块的依赖范围
  Array<Block> GetDependencyScope(Block block);
};

// 内存访问分析
class BufferAccessAnalyzer {
 public:
  // 分析块的读写模式
  Array<BufferRegion> GetReadRegions(Block block);
  Array<BufferRegion> GetWriteRegions(Block block);

  // 检测内存别名
  bool HasMemoryAlias(BufferRegion region1, BufferRegion region2);
};
```

## 5. TIR 变换与优化

### 5.1 基础变换Pass

TVM提供了丰富的TIR变换Pass，位于 `src/tir/transforms/`：

#### 常用变换Pass
```cpp
// 缓冲区扁平化：将多维缓冲区转换为一维
Pass FlattenBuffer();

// 循环变换相关
Pass LoopPartition();      // 循环分区
Pass VectorizeLoop();      // 向量化
Pass UnrollLoop();         // 循环展开

// 内存优化
Pass InjectVirtualThread();    // 虚拟线程注入
Pass InjectPrefetch();         // 预取注入
Pass InjectDoubleBuffer();     // 双缓冲注入

// 设备相关优化
Pass RemoveNoOp();            // 移除无用操作
Pass ConvertBlocksToOpaque(); // 块转换
Pass LowerInitBlock();        // 初始化块降低
```

### 5.2 参数绑定器

ArgBinder负责处理参数绑定和约束检查，位于 `src/tir/transforms/arg_binder.cc`：

```cpp
class ArgBinder {
 public:
  void Bind(const PrimExpr& arg, const PrimExpr& value,
           const std::string& arg_name, bool with_lets = true);

  void BindBuffer(const Buffer& buffer, const Var& var,
                  bool compact_buffer = false);

  std::vector<Stmt> init_nest() const { return init_nest_; }
  std::vector<Var> defs() const { return defs_; }

 private:
  std::unordered_map<const VarNode*, PrimExpr>* def_map_;
  std::vector<Stmt> asserts_;
  std::vector<Stmt> init_nest_;
  std::vector<Var> defs_;
};
```

**核心功能：**

1. **参数匹配**：将形式参数与实际值绑定
2. **约束检查**：验证参数满足必要条件
3. **断言生成**：自动生成运行时检查
4. **变量定义**：管理绑定的变量

## 6. TIR 代码生成

### 6.1 目标代码生成流程

TIR到目标代码的生成流程：

```
TIR → 设备特定优化 → 代码生成 → 目标代码
```

### 6.2 代码生成器接口

```cpp
class CodeGenSourceBase {
 public:
  virtual void AddFunction(const PrimFunc& f) = 0;
  virtual std::string Finish() = 0;
  virtual void PrintStmt(const Stmt& stmt) = 0;
  virtual void PrintExpr(const PrimExpr& expr) = 0;
};

// 具体代码生成器
class CodeGenC : public CodeGenSourceBase;      // C代码生成
class CodeGenLLVM : public CodeGenSourceBase;   // LLVM IR生成
class CodeGenCUDA : public CodeGenSourceBase;   // CUDA代码生成
```

## 7. Python接口设计

### 7.1 Python绑定架构

TIR通过FFI提供Python接口：

```python
# Python中的TIR操作
import tvm
from tvm import tir

# 创建TIR函数
@tir.prim_func
def func(a: tir.handle, b: tir.handle) -> None:
    A = tir.match_buffer(a, (1024, 1024))
    B = tir.match_buffer(b, (1024, 1024))

    for i in tir.serial(1024):
        for j in tir.serial(1024):
            B[i, j] = A[i, j] + 1.0

# 调度操作
s = tir.Schedule(func)
block = s.get_block("root")
i, j = s.get_loops(block)
s.parallel(i)
s.vectorize(j, 128)
```

### 7.2 元编程支持

TIR支持强大的元编程能力：

```python
# 动态生成TIR
def create_gemm(M, N, K):
    @tir.prim_func
    def gemm(a: tir.handle, b: tir.handle, c: tir.handle) -> None:
        A = tir.match_buffer(a, (M, K), "float32")
        B = tir.match_buffer(b, (K, N), "float32")
        C = tir.match_buffer(c, (M, N), "float32")

        for i in tir.serial(M):
            for j in tir.serial(N):
                C[i, j] = 0.0
                for k in tir.serial(K):
                    C[i, j] += A[i, k] * B[k, j]

    return gemm

# 使用动态创建的函数
gemm_func = create_gemm(1024, 1024, 1024)
```

## 8. TIR 与 Halide 的关系

TIR从Halide借鉴了诸多概念，但有重要发展：

### 8.1 相似之处

1. **表达式结构**：基本的表达式和语句类型
2. **调度概念**：循环变换和内存优化
3. **函数式思想**：纯函数式表示

### 8.2 TVM的创新

1. **块抽象**：Block概念提供了更结构化的计算单元
2. **调度状态**：完整的调度状态管理
3. **Python集成**：更深度的Python支持
4. **多设备支持**：更好的设备抽象

## 9. 性能优化技术

### 9.1 内存层次优化

```python
# 缓存优化示例
def optimize_matmul():
    sch = tir.Schedule(matmul_func)

    # 分块
    block = sch.get_block("C")
    i, j, k = sch.get_loops(block)
    i0, i1 = sch.split(i, [32, 32])
    j0, j1 = sch.split(j, [32, 32])
    k0, k1 = sch.split(k, [32, 32])

    # 调整循环顺序
    sch.reorder(i0, j0, k0, i1, j1, k1)

    # 缓存
    A_shared = sch.cache_read(block, 0, "shared")
    B_shared = sch.cache_read(block, 1, "shared")
    C_local = sch.cache_write(block, 0, "local")

    return sch.mod
```

### 9.2 并行化优化

```python
def parallelize_schedule():
    sch = tir.Schedule(func)

    # 并行外层循环
    outer_loops = sch.get_loops(sch.get_block("compute"))[:2]
    sch.parallel(outer_loops[0])

    # 向量化内层循环
    inner_loop = sch.get_loops(sch.get_block("compute"))[2]
    sch.vectorize(inner_loop)

    return sch.mod
```

## 10. 调试与分析工具

### 10.1 调试支持

TVM提供了丰富的调试工具：

```python
# 打印TIR
print(tvm.show_ir(mod))

# 可视化调度
from tvm.contrib import relay_viz
relay_viz.plot_schedule(sch)

# 性能分析
from tvm.autotvm.tuner import XGBTuner
tuner = XGBTuner(func)
```

### 10.2 静态分析工具

```cpp
// 内存使用分析
class MemoryUsageAnalyzer {
 public:
  size_t CalculateTotalAllocated(const PrimFunc& func);
  size_t CalculatePeakMemory(const PrimFunc& func);
  Array<BufferRegion> GetAllocatedBuffers(const PrimFunc& func);
};

// 复杂度分析
class ComplexityAnalyzer {
 public:
  int64_t CalculateArithmeticComplexity(const PrimFunc& func);
  int64_t CalculateMemoryComplexity(const PrimFunc& func);
};
```

## 11. 总结

TIR作为TVM的核心IR，具有以下关键特点：

1. **表达能力强大**：可以精确描述各种张量计算
2. **调度灵活性**：提供丰富的程序变换原语
3. **性能优化友好**：设计充分考虑了性能优化需求
4. **工具链完整**：从分析到代码生成都有完整支持
5. **Python集成良好**：充分利用Python的表达能力

TIR的设计成功地在表达能力和优化潜力之间找到了平衡，为TVM成为强大的机器学习编译器奠定了坚实基础。它的Block抽象概念和调度系统代表了编译器IR设计的前沿水平。

---

*下一篇我们将分析Relax IR，了解TVM在图级别的抽象设计。*