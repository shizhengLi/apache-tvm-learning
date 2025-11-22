# TVM Learning Repository

本仓库专注于 Apache TVM 的学习与研究，提供深入的源码分析、技术博客和实践项目。

## 📚 项目内容

### 📖 技术博客系列
[tech-blogs/](./tech-blogs/) - TVM 源码深度分析技术博客系列

#### 核心架构分析
- **01-TVM架构概览** - TVM整体框架设计与核心理念
- **02-TIR (Tensor IR) 源码深度分析** - 张量级别中间表示详解
- **03-Relax IR 源码深度分析** - 图级别中间表示与高级抽象

#### 编译与优化
- **04-TVM算子编译流程深度解析** - 从高级算子到低级代码的完整流程
- **05-TVM自动调度系统深度解析** - 自动化性能优化技术
- **06-TVM运行时系统深度解析** - 跨平台执行与设备管理

### 🔬 TVM 源码研究
[tvm/](./tvm/) - Apache TVM 完整源码

- **src/** - C++ 核心实现
- **python/** - Python 前端接口
- **include/** - C++ 头文件
- **tests/** - 测试套件
- **docs/** - 官方文档

## 🎯 学习目标

本仓库旨在帮助开发者：

1. **深入理解 TVM 架构** - 从整体设计到具体实现
2. **掌握核心概念** - IR设计、编译优化、自动调度等
3. **提升实践能力** - 通过源码分析理解最佳实践
4. **贡献开源项目** - 为TVM生态系统做出贡献

## 🛠️ 环境要求

### 基础环境
```bash
# Python 环境
Python >= 3.7

# 依赖包
numpy, llvm, cmake, git

# 可选：GPU支持
CUDA >= 10.2 或 ROCm
```

### 构建环境
```bash
# 克隆 TVM 源码
git clone --recursive https://github.com/apache/tvm.git

# 构建
cd tvm
mkdir build
cp cmake/config.cmake build
cd build
cmake ..
make -j8
```

## 📋 学习路径

### 🏃‍♂️ 快速入门
1. 阅读 [TVM架构概览](./tech-blogs/01-TVM架构概览.md)
2. 搭建TVM开发环境
3. 运行第一个TVM示例

### 🚀 进阶学习
1. **IR系统** - TIR 和 Relax IR 深度分析
2. **编译流程** - 理解算子编译和优化过程
3. **自动调度** - 掌握自动化性能优化技术
4. **运行时系统** - 了解执行引擎和设备管理

### 🔬 高级研究
1. **源码贡献** - 参与TVM项目开发
2. **新功能开发** - 扩展TVM功能
3. **性能优化** - 深入性能调优
4. **学术研究** - 基于TVM的研究项目

## 📖 推荐学习资源

### 官方文档
- [TVM 官方文档](https://tvm.apache.org/docs/)
- [TVM 教程](https://tvm.apache.org/docs/tutorial/index.html)
- [TVM API 参考](https://tvm.apache.org/docs/api/index.html)

### 学术论文
- TVM: An Automated End-to-End Optimizing Compiler for Deep Learning
- Ansor: Generating High-Performance Tensor Programs for Deep Learning
- Relax: A Unified IR for Full-Stack Deep Learning

### 社区资源
- [TVM Discuss 论坛](https://discuss.tvm.apache.org/)
- [TVM GitHub](https://github.com/apache/tvm)
- [TVM Community Blog](https://tvm.apache.org/community/blog/)

## 🛠️ 实践项目

### 入级项目
```python
# 1. 自定义算子实现
import tvm
from tvm import te

def create_custom_op():
    # 定义计算
    A = te.placeholder((1024, 1024), name='A')
    B = te.compute((1024, 1024), lambda i, j: A[i, j] * 2.0, name='B')

    # 创建调度
    s = te.create_schedule(B.op)

    # 应用优化
    s[B].parallel(s[B].op.axis[0])

    return s, [A, B]
```

### 进级项目
```python
# 2. 自动调度示例
import tvm
from tvm import auto_scheduler

def auto_schedule_example():
    # 定义计算任务
    M, N, K = 1024, 1024, 1024
    A = te.placeholder((M, K), name='A')
    B = te.placeholder((K, N), name='B')
    k = te.reduce_axis((0, K), name='k')
    C = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k))

    # 自动调度
    task = auto_scheduler.SearchTask(
        func_name="matmul",
        args=[A, B, C],
        target="llvm"
    )

    # 搜索最优配置
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=1000,
        builder=auto_scheduler.LocalBuilder(),
        runner=auto_scheduler.LocalRunner()
    )

    sch, args = auto_scheduler.auto_task_tune(task, tune_option)
    return sch, args
```

## 🧪 测试与验证

### 运行测试
```bash
# 进入TVM目录
cd tvm

# 运行Python测试
python -m pytest tests/python/unittest/test_ir_builder.py

# 运行C++测试
./build/tests/cpp_unittest

# 运行集成测试
python -m pytest tests/python/integration/
```

### 性能基准测试
```python
# 性能测试示例
import tvm
import time
import numpy as np

def benchmark_matmul():
    # 创建测试数据
    M, N, K = 1024, 1024, 1024
    a = np.random.randn(M, K).astype('float32')
    b = np.random.randn(K, N).astype('float32')

    # 构建TVM模块
    # ... 构建过程

    # 性能测试
    start_time = time.time()
    for _ in range(100):
        module.set_input("a", a)
        module.set_input("b", b)
        module.run()

    avg_time = (time.time() - start_time) / 100
    gflops = (2 * M * N * K) / (avg_time * 1e9)
    print(f"Average time: {avg_time:.4f}s, Performance: {gflops:.2f} GFLOPS")
```

## 🤝 贡献指南

### 贡献方式
1. **技术博客改进** - 修正错误、补充内容
2. **代码示例** - 提供更好的实践案例
3. **文档翻译** - 协助完善中文文档
4. **新主题** - 建议新的分析方向

### 提交流程
1. Fork 本仓库
2. 创建功能分支
3. 提交更改
4. 创建 Pull Request

### 代码规范
- 使用中文技术文档
- 提供清晰的代码注释
- 包含完整的示例
- 测试验证

## 📄 许可证

本项目采用 [MIT 许可证](LICENSE)。

## 🙏 致谢

- Apache TVM 社区提供的优秀开源项目
- 所有贡献者付出的努力
- 学习资源作者们的分享

---

**持续更新中...**

欢迎关注仓库更新，也欢迎贡献自己的学习心得和项目经验！

## 📞 联系方式

- **Issues** - 提出问题和建议
- **Discussions** - 技术讨论和交流
- **Pull Request** - 贡献代码和文档

让我们一起深入学习 TVM，推动 AI 编译器技术的发展！