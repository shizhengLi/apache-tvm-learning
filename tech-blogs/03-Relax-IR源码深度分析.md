# Relax IR 源码深度分析

## 1. Relax IR 概述

Relax (Relaxed Expressions) 是TVM的高级图级别中间表示(IR)，专为现代深度学习工作负载设计。它提供了一个函数式、静态类型的IR，支持动态shape、符号执行和渐进式类型推导。

### 1.1 Relax的核心设计理念

1. **函数式编程范式**：表达式无副作用，便于优化和分析
2. **渐进式类型系统**：从动态类型到静态类型的平滑过渡
3. **符号执行支持**：支持动态shape和符号变量
4. **张量抽象**：原生支持张量操作和shape计算
5. **跨层级优化**：与TIR无缝集成，支持端到端优化

### 1.2 在TVM架构中的位置

```
前端框架 → Relax IR → 优化Pass → TIR → 目标代码
          ↑
    动态shape支持
```

## 2. 核心数据结构

### 2.1 表达式系统 (Expressions)

Relax的表达式系统位于 `include/tvm/relax/expr.h`：

```cpp
// 基础表达式类型
class RelaxExprNode : public BaseExprNode {
 public:
  mutable Span span;  // 源码位置信息
  // ...
};

// 变量系统
class IdNode : public Object {
 public:
  ffi::String name_hint;  // 变量名提示
  // 注意：Id是唯一的，用于区分不同的变量实例
};

class VarNode : public RelaxExprNode {
 public:
  Id vid;                    // 唯一标识符
  Optional<StructInfo> struct_info_;  // 结构信息（类型+shape）
  // ...
};

// 字面量
class ConstantNode : public RelaxExprNode {
 public:
  runtime::NDArray data;  // 存储常量数据
};

// 元组
class TupleNode : public RelaxExprNode {
 public:
  ffi::Array<Expr> fields;  // 元组元素
};

// 函数调用
class CallNode : public RelaxExprNode {
 public:
  Expr op;                       // 操作符
  ffi::Array<Expr> args;         // 参数
  ffi::Map<String, ObjectRef> attrs;  // 属性
  Optional<StructInfo> sinfo_args;    // 参数结构信息
  Optional<Expr> op_dtype;             // 操作符数据类型
};

// 函数定义
class FunctionNode : public BaseFuncNode {
 public:
  Optional<Expr> ret_struct_info;  // 返回结构信息
  ffi::Array<Var> params;          // 参数
  Expr body;                       // 函数体
  bool is_pure;                    // 是否纯函数
  bool is_entry;                   // 是否入口函数
  ffi::Map<String, ObjectRef> attrs;  // 属性
  Optional<DictAttrs> struct_info_;   // 结构信息
};
```

### 2.2 结构信息系统 (StructInfo)

StructInfo是Relax的核心创新，它结合了静态类型信息和运行时结构信息：

```cpp
// 基础结构信息
class StructInfoNode : public Object {
 public:
  mutable Span span;
  // ...
};

// 对象结构信息
class ObjectStructInfoNode : public StructInfoNode {
  // 通用对象，类型信息最模糊
};

// 基础值结构信息
class PrimStructInfoNode : public StructInfoNode {
 public:
  ffi::Optional<PrimExpr> value;  // 基础值（如果已知）
  DataType dtype;                 // 数据类型
};

// Shape结构信息
class ShapeStructInfoNode : public StructInfoNode {
 public:
  int ndim;                       // 维度数
  Optional<Array<PrimExpr>> values;  // shape值（如果已知）
};

// 张量结构信息
class TensorStructInfoNode : public StructInfoNode {
 public:
  Optional<Array<PrimExpr>> shape;  // 张量shape
  DataType dtype;                   // 数据类型
  Optional<String> vdevice;         // 设备信息
};

// 元组结构信息
class TupleStructInfoNode : public StructInfoNode {
 public:
  Array<StructInfo> fields;         // 元组字段的结构信息
};

// 函数结构信息
class FuncStructInfoNode : public StructInfoNode {
 public:
  Optional<Array<StructInfo>> params;  // 参数结构信息
  StructInfo ret;                      // 返回结构信息
  bool is_pure;                        // 是否纯函数
  bool is_derived;                     // 是否派生类型
};
```

**StructInfo的关键特点：**

1. **渐进式类型推导**：从最宽松的ObjectStructInfo到精确的TensorStructInfo
2. **Shape信息保留**：支持符号shape表达式
3. **运行时检查**：可以生成运行时类型/shape验证
4. **优化指导**：为编译器提供丰富的优化信息

### 2.3 绑定系统 (Bindings)

Relax使用绑定来表示值与变量之间的关系：

```cpp
// 变量绑定
class VarBindingNode : public BindingNode {
 public:
  Var var;     // 目标变量
  Expr value;  // 绑定的值
};

// 绑定块
class DataflowBlockNode : public BindingBlockNode {
 public:
  Array<Binding> bindings;  // 绑定列表
  bool can_be_inlined;      // 是否可内联
};

class BindingBlockNode : public RelaxExprNode {
 public:
  Array<Binding> bindings;  // 绑定列表
};
```

## 3. BlockBuilder 系统

### 3.1 BlockBuilder的核心设计

BlockBuilder是构建Relax程序的核心工具，位于 `include/tvm/relax/block_builder.h`：

```cpp
class BlockBuilderNode : public Object {
 public:
  // 全局上下文管理
  virtual NameSupply name_supply() = 0;
  virtual IRModule GetContextIRModule() const = 0;
  virtual void UpdateContextIRModule(IRModule new_module) = 0;

  // 作用域管理
  virtual void BeginBindingBlock() = 0;
  virtual void BeginDataflowBlock() = 0;
  virtual BindingBlock EndBlock() = 0;

  // 表达式发射
  virtual Var Emit(Expr expr, Optional<StructInfo> sinfo = NullOpt) = 0;
  virtual Var EmitMatchCast(Expr value, StructInfo struct_info) = 0;

  // 规范化
  virtual Expr Normalize(Expr expr) = 0;
  virtual StructInfo GetStructInfo(Expr expr) = 0;
  virtual void SetStructInfo(Expr expr, StructInfo sinfo) = 0;
};
```

### 3.2 BlockBuilder的功能分类

#### 1. 全局上下文管理
```cpp
// 管理IRModule，提供全局查询和更新
IRModule module = builder->GetContextIRModule();
builder->UpdateContextIRModule(updated_module);
```

#### 2. 作用域管理
```cpp
// 创建作用域
builder->BeginDataflowBlock();
var = builder->Emit(some_expr);
BindingBlock block = builder->EndBlock();
```

#### 3. 规范化系统
```cpp
// 自动推导类型和shape
Expr normalized = builder->Normalize(expr);
StructInfo sinfo = builder->GetStructInfo(expr);
```

### 3.3 规范化流程

规范化是Relax的关键机制，它确保表达式处于标准形式：

```cpp
Expr BlockBuilder::Normalize(Expr expr) {
  // 1. 常量折叠
  if (auto* constant = expr.as<ConstantNode>()) {
    return FoldConstant(constant);
  }

  // 2. 函数调用规范化
  if (auto* call = expr.as<CallNode>()) {
    return NormalizeCall(call);
  }

  // 3. 元组规范化
  if (auto* tuple = expr.as<TupleNode>()) {
    return NormalizeTuple(tuple);
  }

  // ... 其他规范化规则
}
```

## 4. 分析框架

### 4.1 结构信息推导

Relax提供了强大的结构信息推导系统，位于 `src/relax/analysis/struct_info_analysis.cc`：

```cpp
class StaticTypeDeriver : public StructInfoFunctor<Type(const StructInfo&)> {
 public:
  Type VisitStructInfo_(const ObjectStructInfoNode* op) final {
    return ObjectType(op->span);
  }

  Type VisitStructInfo_(const PrimStructInfoNode* op) final {
    return PrimType(op->dtype, op->span);
  }

  Type VisitStructInfo_(const TensorStructInfoNode* op) final {
    return TensorType(op->ndim, op->dtype);
  }

  Type VisitStructInfo_(const TupleStructInfoNode* op) final {
    Array<Type> fields = op->fields.Map([this](const StructInfo& sinfo) {
      return this->VisitStructInfo(sinfo);
    });
    return TupleType(fields, op->span);
  }
};
```

### 4.2 形状分析

形状分析是Relax的重要特性，支持动态shape：

```cpp
// 形状推导器
class ShapeAnalyzer : public ExprFunctor<Optional<Array<PrimExpr>>(Expr)> {
 public:
  Optional<Array<PrimExpr>> VisitExpr_(const VarNode* op) final {
    return GetShapeFromStructInfo(op->struct_info_);
  }

  Optional<Array<PrimExpr>> VisitExpr_(const CallNode* op) final {
    // 根据操作符推导输出shape
    if (auto tir_op = op->op.as<GlobalVar>()) {
      return DeriveShapeFromTIR(tir_op, op->args);
    }
    // ... 其他操作符的shape推导
  }
};
```

### 4.3 数据流分析

Relax提供了完整的数据流分析框架：

```cpp
// 使用定义链分析
UDChain AnalyzeUDChain(const IRModule& module);

// 计算图分区
Array<IRModule> GraphPartition(const IRModule& module,
                               PartitionRule rule);

// 内存分析
MemoryPlan AnalyzeMemoryUsage(const IRModule& module);
```

## 5. 变换系统 (Transformations)

### 5.1 变换框架

Relax的变换系统位于 `src/relax/transform/`，提供了丰富的优化pass：

```cpp
// 基础变换接口
class TransformPass {
 public:
  virtual IRModule operator()(IRModule mod) const = 0;
};

// 函数级别变换
IRModule TransformFunction(IRModule mod,
                          std::function<Function(Function)> f);
```

### 5.2 核心优化Pass

#### 常量折叠
```cpp
// 位于 src/relax/transform/fold_constant.cc
IRModule FoldConstant(IRModule mod) {
  class ConstantFolder : public ExprMutator {
    Expr VisitExpr_(const CallNode* op) final {
      // 尝试计算常量表达式
      if (CanFoldConstant(op)) {
        return EvaluateConstantOp(op);
      }
      return ExprMutator::VisitExpr_(op);
    }
  };

  return PostOrderRewrite(mod, ConstantFolder());
}
```

#### 算子融合
```cpp
// 位于 src/relax/transform/fuse_ops.cc
IRModule FuseOps(IRModule mod, const FuseOpsConfig& config) {
  class OpFusion : public ExprMutator {
    Expr VisitExpr_(const CallNode* op) final {
      // 检查是否可以与前驱算子融合
      if (CanFuseWithPrevious(op)) {
        return CreateFusedOperator(op);
      }
      return ExprMutator::VisitExpr_(op);
    }
  };
}
```

#### 死代码消除
```cpp
// 位于 src/relax/transform/dead_code_elimination.cc
IRModule DeadCodeElimination(IRModule mod) {
  class DCE : public ExprMutator {
   public:
    Expr VisitExpr_(const VarNode* op) final {
      // 检查变量是否被使用
      if (IsUnused(op)) {
        return Expr(); // 删除未使用的变量
      }
      return ExprMutator::VisitExpr_(op);
    }
  };
}
```

#### 函数内联
```cpp
// 位于 src/relax/transform/inline_functions.cc
IRModule InlineFunctions(IRModule mod) {
  class FunctionInliner : public ExprMutator {
    Expr VisitExpr_(const CallNode* op) final {
      if (ShouldInline(op->op)) {
        auto func = GetFunction(op->op);
        return InlineFunctionCall(func, op->args);
      }
      return ExprMutator::VisitExpr_(op);
    }
  };
}
```

### 5.3 高级变换

#### 自动微分
```cpp
// 位于 src/relax/transform/gradient.cc
IRModule Gradient(IRModule mod, const String& func_name) {
  class AutoDiff : public ExprMutator {
    // 实现反向模式自动微分
    Expr VisitExpr_(const CallNode* op) final {
      return GenerateBackwardPass(op);
    }
  };
}
```

#### 内存规划
```cpp
// 位于 src/relax/transform/static_plan_block_memory.cc
IRModule StaticPlanBlockMemory(IRModule mod) {
  return PlanMemoryAllocation(mod);
}
```

#### 精度混合
```cpp
// 位于 src/relax/transform/to_mixed_precision.cc
IRModule ToMixedPrecision(IRModule mod, const MixedPrecisionConfig& config) {
  return CastToMixedPrecision(mod, config);
}
```

## 6. 前端集成

### 6.1 框架导入器

Relax支持从多种深度学习框架导入模型：

```python
# TensorFlow导入
import tvm
from tvm import relax
from tvm.relax.frontend import tensorflow

model = tensorflow.from_tensorflow(saved_model_dir)

# PyTorch导入
from tvm.relax.frontend import pytorch

model = pytorch.from_pytorch(torch_model, input_info)

# ONNX导入
from tvm.relax.frontend import onnx

model = onnx.from_onnx(onnx_model)
```

### 6.2 模型构建API

```python
# 使用BlockBuilder构建模型
import tvm
from tvm import relax
from tvm.relax.testing import nn

def build_mlp():
    bb = relax.BlockBuilder()

    with bb.function("main", (nn.Tensor([128, 784], "float32"),)):
        x = bb.match_cast(bb.arg("x"), relax.TensorStructInfo([128, 784], "float32"))

        # 线性层
        w1 = bb.emit(nn.init.zeros([784, 256], "float32"))
        b1 = bb.emit(nn.init.zeros([256], "float32"))
        x1 = bb.emit(relax.op.nn.linear(x, w1, b1))
        x1 = bb.emit(relax.op.nn.relu(x1))

        # 输出层
        w2 = bb.emit(nn.init.zeros([256, 10], "float32"))
        b2 = bb.emit(nn.init.zeros([10], "float32"))
        out = bb.emit(relax.op.nn.linear(x1, w2, b2))

        bb.emit_func_output(out)

    return bb.get()
```

## 7. 与TIR的集成

### 7.1 CallTIR操作

Relax通过CallTIR操作与TIR集成：

```python
# 直接调用TIR函数
@tir.prim_func
def matmul_tir(a: tir.handle, b: tir.handle, c: tir.handle) -> None:
    # TIR实现
    pass

# 在Relax中调用
def relax_call_tir(a, b):
    tir_func = tir.extern("matmul_tir", [a, b])
    return relax.call_tir(tir_func, [a, b])
```

### 7.2 渐进式降低

Relax支持渐进式将高级操作降低为TIR：

```cpp
// 算子合法化
IRModule LegalizeOps(IRModule mod, Target target) {
  class Legalizer : public ExprMutator {
    Expr VisitExpr_(const CallNode* op) final {
      // 将高级操作转换为TIR原语
      if (IsHighLevelOp(op)) {
        return ConvertToTIR(op, target_);
      }
      return ExprMutator::VisitExpr_(op);
    }
  };
}
```

## 8. 数据流优化

### 8.1 DataflowRegion

Relax支持显式的数据流区域优化：

```cpp
class DataflowBlockNode : public BindingBlockNode {
 public:
  Array<Binding> bindings;
  bool can_be_inlined = true;  // 数据流块可以内联优化
};
```

```python
# 创建数据流区域
with bb.dataflow():
    # 这里的绑定可以自动内联
    x = bb.emit(op1(input))
    y = bb.emit(op2(x))  # x可以被内联到y中
    z = bb.emit(op3(y))  # y可以被内联到z中
```

### 8.2 内存优化

```python
# 就地操作优化
def inplace_optimize():
    bb = relax.BlockBuilder()

    with bb.function("main"):
        x = bb.match_cast(bb.arg("x"), ...)

        with bb.dataflow():
            # 标记为就地操作
            x_updated = bb.emit(relax.op.add(x, 1.0), inplace=True)

        bb.emit_func_output(x_updated)
```

## 9. 类型系统

### 9.1 静态类型

Relax提供了强类型的静态类型系统：

```python
# 类型注解
def typed_function(x: Tensor((1, 3, 224, 224), "float32")) -> Tensor((1000,), "float32"):
    # 实现有类型检查
    return model(x)
```

### 9.2 动态类型

同时也支持动态类型：

```python
# 动态类型函数
def dynamic_function(x):
    # 类型在运行时检查
    return process(x)
```

### 9.3 类型推导

```python
# 自动类型推导
def auto_typed_function(x):
    # x的类型会被自动推导
    y = relax.op.add(x, 1.0)  # y类型从add操作推导
    return y  # 函数返回类型从y推导
```

## 10. 错误处理与诊断

### 10.1 类型错误

```python
# 类型错误会在编译时捕获
try:
    mod = relax.transform.CheckTypes()(mod)
except tvm.TVMError as e:
    print(f"Type error: {e}")
```

### 10.2 Shape错误

```python
# Shape检查
def check_shapes():
    bb = relax.BlockBuilder()

    with bb.function("main"):
        x = bb.match_cast(bb.arg("x"), relax.TensorStructInfo([None, 784], "float32"))
        # 如果输入shape不匹配[*, 784]，会在运行时报错
```

## 11. 性能优化

### 11.1 编译时优化

```python
# 应用优化序列
optimized_mod = tvm.transform.Sequential([
    relax.transform.FoldConstant(),
    relax.transform.FuseOps(),
    relax.transform.DeadCodeElimination(),
    relax.transform.LegalizeOps(),
])(mod)
```

### 11.2 运行时优化

```python
# JIT编译
ex = relax.build(mod, target="llvm")
ex.set_input("input", data)
ex.run()
```

## 12. 调试工具

### 12.1 可视化

```python
# 打印Relax IR
print(mod.show())

# 可视化数据流
from tvm.contrib import graph_executor
graph_executor.create(mod, target).debug()
```

### 12.2 分析工具

```python
# 性能分析
from tvm.relax.analysis import estimate_flops

flops = estimate_flops(mod)
print(f"Model FLOPs: {flops}")

# 内存分析
memory_usage = analyze_memory(mod)
print(f"Peak memory: {memory_usage}")
```

## 13. 总结

Relax IR作为TVM的高级IR，具有以下关键特性：

1. **函数式设计**：无副作用，便于优化
2. **渐进式类型**：从动态到静态的平滑过渡
3. **符号执行**：原生支持动态shape
4. **结构信息**：丰富的类型和shape信息
5. **与TIR集成**：无缝的端到端优化
6. **数据流优化**：显式的数据流区域支持
7. **现代语言特性**：元组、高阶函数等

Relax的设计代表了深度学习编译器的最新发展方向，它成功地平衡了表达能力和优化潜力，为TVM处理现代深度学习工作负载提供了强大支持。其创新的StructInfo系统和BlockBuilder架构为深度学习编译器设计提供了新的思路。

---

*下一篇我们将分析TVM的算子编译流程，了解从高级IR到可执行代码的完整过程。*