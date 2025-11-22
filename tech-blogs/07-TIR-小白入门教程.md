# TIR (Tensor IR) 小白入门教程

## 🎯 什么是TIR？

想象一下，你有一个神经网络模型，就像一份菜谱。TIR (Tensor Intermediate Representation) 就是把这份菜谱翻译成计算机能理解的"低级语言"的过程。

### 🍳 生活中的比喻

假设你要做一个"番茄炒蛋"：
- **原始想法**："我想吃番茄炒蛋" (高级指令)
- **菜谱步骤**：
  1. 洗番茄、切番茄
  2. 打鸡蛋、加调料
  3. 热锅、下油
  4. 炒鸡蛋、盛起
  5. 炒番茄、加鸡蛋
  6. 调味、装盘
- **TIR角色**：就是把菜谱步骤翻译成厨师能精确执行的具体动作

## 💡 为什么需要TIR？

### 1. 跨平台兼容性
```
你的神经网络代码 (Python/PyTorch)
         ↓
      TVM处理
         ↓
    TIR表示 (统一中间表示)
         ↓
    ├─ CPU版本
    ├─ GPU版本
    ├─ 手机版本
    └─ 其他设备版本
```

### 2. 性能优化
- **Python代码**：简单易懂，但运行慢
- **TIR代码**：稍复杂，但运行快

就像：
- **话指挥**：简单，但不如直接演奏精确
- **五线谱**：需要学习，但能完美表达音乐细节

## 🏗️ TIR的基本概念

### 1. 张量 (Tensor)
```python
# 普通Python
matrix = [[1, 2], [3, 4]]  # 这只是列表的列表

# TIR中的张量
# 有明确的数据类型、内存布局、计算方式
```

### 2. 缓冲区 (Buffer)
```python
# TIR中的缓冲区定义
A = Buffer(shape=[1024, 1024], dtype="float32", name="A")
# 就像是一个有明确容量和用途的容器
```

### 3. 计算块 (Block)
```python
# 一个计算块就像一个独立的工作站
with tir.block([1024, 1024], "C") as [i, j]:
    C[i, j] = A[i, j] + B[i, j]  # 这里的每个元素计算
```

## 🧩 TIR的核心组件

### 1. 函数 (PrimFunc)
```python
import tvm
from tvm import tir

@tir.prim_func  # 这是一个TIR函数
def add_tensors(a: tir.handle, b: tir.handle, c: tir.handle):
    # 声明输入和输出缓冲区
    A = tir.match_buffer(a, [1024, 1024], "float32")
    B = tir.match_buffer(b, [1024, 1024], "float32")
    C = tir.match_buffer(c, [1024, 1024], "float32")

    # 计算逻辑
    for i in tir.serial(1024):      # 外层循环
        for j in tir.serial(1024):  # 内层循环
            C[i, j] = A[i, j] + B[i, j]  # 逐元素相加
```

**解释：**
- `@tir.prim_func` - 告诉TVM这是一个底层函数
- `tir.handle` - 内存句柄，指向实际数据
- `tir.match_buffer` - 匹配输入数据到缓冲区
- `tir.serial(1024)` - 顺序循环（可以并行化优化）

### 2. 循环 (Loops)
```python
# 不同类型的循环

# 1. 串行循环（默认）
for i in tir.serial(1024):
    # 一个接一个执行

# 2. 并行循环
for i in tir.parallel(1024):
    # 可以同时执行多个

# 3. 向量化循环
for i in tir.vectorized(1024):
    # 一次性处理多个元素（CPU SSE/AVX指令）

# 4. 展开循环
for i in tir.unroll(1024):
    # 把循环展开成重复代码，减少分支开销
```

### 3. 内存访问
```python
# 缓冲区读写
value = A[i, j]        # 读取缓冲区元素
C[i, j] = value + 1    # 写入缓冲区元素

# 内存类型
shared_buffer = tir.alloc_buffer([1024, 1024], "float32", scope="shared")
# shared: GPU共享内存
# local:  本地寄存器
# global: 全局内存
```

## 🚀 TIR优化入门

### 1. 简单的矩阵乘法

**基础版本：**
```python
@tir.prim_func
def matmul_basic(a: tir.handle, b: tir.handle, c: tir.handle):
    A = tir.match_buffer(a, [1024, 1024], "float32")
    B = tir.match_buffer(b, [1024, 1024], "float32")
    C = tir.match_buffer(c, [1024, 1024], "float32")

    for i in tir.serial(1024):
        for j in tir.serial(1024):
            for k in tir.serial(1024):
                C[i, j] += A[i, k] * B[k, j]
```

### 2. 优化版本1 - 分块 (Blocking)
```python
@tir.prim_func
def matmul_block(a: tir.handle, b: tir.handle, c: tir.handle):
    A = tir.match_buffer(a, [1024, 1024], "float32")
    B = tir.match_buffer(b, [1024, 1024], "float32")
    C = tir.match_buffer(c, [1024, 1024], "float32")

    # 分块计算，提高缓存命中率
    for i0 in tir.serial(32):        # 外层分块
        for j0 in tir.serial(32):
            for i1 in tir.serial(32):    # 内层循环
                for j1 in tir.serial(32):
                    for k in tir.serial(1024):
                        i = i0 * 32 + i1
                        j = j0 * 32 + j1
                        C[i, j] += A[i, k] * B[k, j]
```

### 3. 优化版本2 - 并行化
```python
@tir.prim_func
def matmul_parallel(a: tir.handle, b: tir.handle, c: tir.handle):
    A = tir.match_buffer(a, [1024, 1024], "float32")
    B = tir.match_buffer(b, [1024, 1024], "float32")
    C = tir.match_buffer(c, [1024, 1024], "float32")

    # 外层循环并行化
    for i0 in tir.parallel(32):      # 注意这里是parallel
        for j0 in tir.serial(32):
            for i1 in tir.serial(32):
                for j1 in tir.serial(32):
                    for k in tir.serial(1024):
                        i = i0 * 32 + i1
                        j = j0 * 32 + j1
                        C[i, j] += A[i, k] * B[k, j]
```

## 🛠️ TIR调度器 (Scheduler)

调度器就像一个"智能助手"，帮你自动优化TIR代码：

```python
import tvm
from tvm import tir

# 1. 创建基础函数
@tir.prim_func
def matmul_func(a: tir.handle, b: tir.handle, c: tir.handle):
    A = tir.match_buffer(a, [1024, 1024], "float32")
    B = tir.match_buffer(b, [1024, 1024], "float32")
    C = tir.match_buffer(c, [1024, 1024], "float32")

    for i in tir.serial(1024):
        for j in tir.serial(1024):
            C[i, j] = 0.0
            for k in tir.serial(1024):
                C[i, j] += A[i, k] * B[k, j]

# 2. 创建调度器
sch = tir.Schedule(matmul_func)

# 3. 获取计算块和循环
block = sch.get_block("C")
i, j, k = sch.get_loops(block)

# 4. 应用优化
# 分块
i0, i1 = sch.split(i, [32, 32])
j0, j1 = sch.split(j, [32, 32])
k0, k1 = sch.split(k, [32, 32])

# 调整循环顺序
sch.reorder(i0, j0, k0, i1, j1, k1)

# 并行化
sch.parallel(i0)

# 向量化
sch.vectorize(k1)

# 获取优化后的函数
optimized_func = sch.mod["main"]
```

## 🎨 TIR与Python的对比

### Python代码：
```python
def matrix_add(A, B):
    C = []
    for i in range(len(A)):
        row = []
        for j in range(len(A[0])):
            row.append(A[i][j] + B[i][j])
        C.append(row)
    return C
```

### 对应的TIR代码：
```python
@tir.prim_func
def matrix_add_tir(a: tir.handle, b: tir.handle, c: tir.handle):
    A = tir.match_buffer(a, [1024, 1024], "float32")
    B = tir.match_buffer(b, [1024, 1024], "float32")
    C = tir.match_buffer(c, [1024, 1024], "float32")

    for i in tir.serial(1024):
        for j in tir.serial(1024):
            C[i, j] = A[i, j] + B[i, j]
```

**主要区别：**
1. **类型明确**：TIR要指定数据类型
2. **内存明确**：TIR要明确内存布局
3. **性能导向**：TIR设计考虑硬件特性
4. **优化空间**：TIR提供了更多优化可能

## 🔍 TIR调试技巧

### 1. 打印TIR代码
```python
# 查看生成的TIR代码
print(tvm.lower(sch.mod["main"], [], simple_mode=False))
```

### 2. 可视化计算图
```python
# 使用TVM提供的可视化工具
from tvm.contrib import graph_executor

# 构建和运行
mod = tvm.build(optimized_func, target="llvm")
dev = tvm.cpu(0)
module = graph_executor.GraphModule(mod["default"](dev))
```

### 3. 性能分析
```python
import time

# 性能测试
start_time = time.time()
module.run()
end_time = time.time()

print(f"执行时间: {end_time - start_time:.4f}秒")
```

## 📚 TIR学习路线

### 第1步：理解基本概念
- 什么是张量、缓冲区、计算块
- TIR与Python的区别
- 基本的TIR语法

### 第2步：练习简单例子
- 向量加法、矩阵乘法
- 循环优化
- 内存访问模式

### 第3步：学习调度器
- 手动调度
- 自动调度
- 调度策略选择

### 第4步：深入优化技术
- 内存布局优化
- 并行化技术
- 设备特定优化

## 🎯 实践练习

### 练习1：向量加法
```python
# 尝试实现一个向量加法的TIR函数
@tir.prim_func
def vector_add(a: tir.handle, b: tir.handle, c: tir.handle):
    # TODO: 实现128维向量的加法
    pass
```

### 练习2：矩阵转置
```python
# 尝试实现矩阵转置的TIR函数
@tir.prim_func
def matrix_transpose(a: tir.handle, c: tir.handle):
    # TODO: 实现矩阵转置
    pass
```

### 练习3：性能优化
```python
# 对基础函数进行优化
def optimize_function(func):
    # TODO: 添加并行化、向量化等优化
    pass
```

## 🤔 常见问题

### Q1: 什么时候需要学习TIR？
**A**: 当你需要：
- 深入理解AI编译器工作原理
- 自定义高性能算子
- 优化现有模型性能
- 贡献TVM项目

### Q2: TIR很难学吗？
**A**: 不难！就像学习任何新语言：
- 从简单例子开始
- 多动手实践
- 理解核心概念
- 逐步深入学习

### Q3: 必须写TIR代码吗？
**A**: 不一定！
- **简单使用**：TVM自动处理
- **性能调优**：可能需要写TIR
- **自定义算子**：必须了解TIR

## 🎉 总结

TIR就像是AI编译器的"汇编语言"：

- **作用**：连接高级深度学习代码和底层硬件执行
- **特点**：精确控制，高性能
- **学习**：从简单开始，逐步深入
- **应用**：性能优化、自定义算子、硬件适配

掌握TIR，你就掌握了AI模型高性能部署的核心技能！

---

**下一步：**
1. 动手实践基础例子
2. 尝试简单的性能优化
3. 阅读更多TVM官方文档
4. 参与TVM社区讨论

记住：**实践是最好的学习方式！** 🚀