# TVM自动搜索最佳调度实现原理深度解析

## 🎯 引言：为什么要自动搜索调度？

想象一下，你有一个数学题：
```
计算：sum(A[i] * B[i] for i in range(1000000))
```

### 不同的计算方式
```python
# 方式1：简单循环
result = 0
for i in range(1000000):
    result += A[i] * B[i]

# 方式2：并行计算
# 分成8个线程，每个线程计算1/8的数据
result1 = sum(A[i] * B[i] for i in range(0, 125000))
result2 = sum(A[i] * B[i] for i in range(125000, 250000))
# ...
result = result1 + result2 + ... + result8

# 方式3：向量化计算
# 一次性处理4个或8个数据
# 使用CPU的SIMD指令

# 方式4：分块计算
# 把大数组分成小块，提高缓存命中率
```

**问题**：哪种方式最快？答案取决于：
- CPU架构（多少核心？是否有SIMD？缓存大小？）
- 数据大小和特点
- 内存带宽和延迟

**传统解决方法**：专家手动尝试各种组合
**现代解决方法**：TVM自动搜索最佳方案！

## 🔍 自动调度器的工作流程

### 整体架构图
```
原始计算 (如矩阵乘法)
    ↓
搜索空间生成 (所有可能的优化方案)
    ↓
搜索算法 (遗传算法、模拟退火等)
    ↓
性能评估 (实际运行测量)
    ↓
最优方案选择
    ↓
生成高效代码
```

### 核心步骤详解

#### 1. 搜索空间生成
```python
# TVM如何生成"所有可能的方案"？

def generate_search_space(computation):
    search_space = []

    # 循环变换
    loop_transforms = [
        ("split", [2, 4, 8, 16, 32]),      # 分块大小
        ("reorder", various_orders),       # 循环顺序
        ("parallel", [True, False]),        # 是否并行
        ("vectorize", [4, 8, 16]),         # 向量化长度
        ("unroll", [0, 1, 2, 4])           # 展开程度
    ]

    # 内存布局
    memory_layouts = [
        ("row_major", "col_major"),        # 行主序/列主序
        ("shared", "global", "local")      # GPU内存类型
    ]

    # 缓存策略
    cache_strategies = [
        ("cache_read", ["A", "B"]),        # 缓存读取
        ("cache_write", ["C"]),            # 缓存写入
        ("reuse_buffer", ["temp"])         # 缓冲区复用
    ]

    # 生成所有组合 (实际中会智能剪枝)
    return combine_all_options(loop_transforms, memory_layouts, cache_strategies)
```

#### 2. 搜索算法实现

**遗传算法示例：**
```python
class GeneticTuning:
    def __init__(self, population_size=50, generations=100):
        self.population_size = population_size
        self.generations = generations

    def evolve(self, search_space):
        # 初始化种群
        population = self.initialize_population(search_space)

        for generation in range(self.generations):
            # 评估适应度（性能）
            fitness_scores = [self.evaluate_fitness(individual)
                            for individual in population]

            # 选择优秀个体
            selected = self.selection(population, fitness_scores)

            # 交叉（组合优秀方案）
            offspring = self.crossover(selected)

            # 变异（随机改变）
            offspring = self.mutation(offspring)

            # 更新种群
            population = offspring

        return self.get_best_individual(population)

    def evaluate_fitness(self, schedule):
        """评估一个调度的性能"""
        try:
            # 生成代码
            compiled_code = self.compile_schedule(schedule)

            # 运行并测量时间
            execution_time = self.benchmark(compiled_code)

            # 适应度 = 1/执行时间（越快越好）
            return 1.0 / execution_time

        except Exception:
            # 编译失败，适应度为0
            return 0.0
```

#### 3. 性能测量系统

```python
class PerformanceMeasurer:
    def __init__(self, warmup_trials=5, measure_trials=10):
        self.warmup_trials = warmup_trials
        self.measure_trials = measure_trials

    def measure_schedule(self, schedule, input_data):
        """测量调度性能"""
        times = []

        try:
            # 编译调度
            compiled = self.compile_schedule(schedule)

            # 预热（避免首次执行的冷启动开销）
            for _ in range(self.warmup_trials):
                compiled.run(input_data)

            # 正式测量
            for _ in range(self.measure_trials):
                start_time = time.perf_counter()
                compiled.run(input_data)
                end_time = time.perf_counter()
                times.append(end_time - start_time)

            # 返回平均时间（去掉最快和最慢的）
            times.sort()
            if len(times) >= 3:
                times = times[1:-1]  # 去掉极端值

            return sum(times) / len(times)

        except Exception as e:
            print(f"测量失败: {e}")
            return float('inf')  # 表示极差性能
```

## 🧬 深入搜索算法实现

### 1. 遗传算法 (Genetic Algorithm)

```python
import random
import numpy as np

class ScheduleGenome:
    """调度方案的"基因"表示"""

    def __init__(self, genes=None):
        if genes is None:
            self.genes = self.random_genes()
        else:
            self.genes = genes

    def random_genes(self):
        """生成随机调度基因"""
        return {
            'tile_sizes': random.choice([2, 4, 8, 16, 32, 64]),
            'parallel_dims': random.sample(range(6), 2),  # 选择2个维度并行
            'vectorize_dim': random.randint(0, 2),       # 向量化哪个维度
            'unroll_factor': random.choice([0, 2, 4, 8]),
            'reorder': list(np.random.permutation([0, 1, 2]))  # 循环顺序
        }

class GeneticScheduler:
    def __init__(self, population_size=100, mutation_rate=0.1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elite_size = 5  # 保留最好的个体数量

    def optimize(self, search_space, generations=50):
        """遗传算法主流程"""
        # 1. 初始化种群
        population = [ScheduleGenome() for _ in range(self.population_size)]

        best_genome = None
        best_fitness = 0

        for generation in range(generations):
            # 2. 评估适应度
            fitness_scores = []
            for genome in population:
                fitness = self.evaluate_fitness(genome)
                fitness_scores.append(fitness)

                # 记录最佳个体
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_genome = genome

            print(f"Generation {generation}: Best fitness = {best_fitness}")

            # 3. 选择（轮盘赌选择）
            selected = self.roulette_selection(population, fitness_scores)

            # 4. 交叉
            offspring = self.crossover(selected)

            # 5. 变异
            offspring = self.mutation(offspring)

            # 6. 精英保留（保留最好的个体）
            elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
            for i, idx in enumerate(elite_indices):
                offspring[i] = population[idx]

            population = offspring

        return best_genome

    def evaluate_fitness(self, genome):
        """评估基因适应度（实际性能）"""
        try:
            # 根据基因生成具体调度
            schedule = self.genome_to_schedule(genome)

            # 编译和测试
            execution_time = self.benchmark_schedule(schedule)

            # 适应度 = 1 / 执行时间（时间越短，适应度越高）
            return 1.0 / execution_time

        except Exception:
            return 0.0  # 编译失败或运行错误

    def crossover(self, parents):
        """交叉操作：组合两个父代的基因"""
        offspring = []
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                parent1, parent2 = parents[i], parents[i + 1]

                # 单点交叉
                crossover_point = random.randint(1, len(parent1.genes) - 1)
                genes1 = {}
                genes2 = {}

                gene_keys = list(parent1.genes.keys())
                for i, key in enumerate(gene_keys):
                    if i < crossover_point:
                        genes1[key] = parent1.genes[key]
                        genes2[key] = parent2.genes[key]
                    else:
                        genes1[key] = parent2.genes[key]
                        genes2[key] = parent1.genes[key]

                offspring.extend([
                    ScheduleGenome(genes1),
                    ScheduleGenome(genes2)
                ])
        return offspring

    def mutation(self, population):
        """变异操作：随机改变基因"""
        for genome in population:
            if random.random() < self.mutation_rate:
                # 随机选择一个基因进行变异
                gene_key = random.choice(list(genome.genes.keys()))

                if gene_key == 'tile_sizes':
                    genome.genes[gene_key] = random.choice([2, 4, 8, 16, 32, 64])
                elif gene_key == 'unroll_factor':
                    genome.genes[gene_key] = random.choice([0, 2, 4, 8])
                # ... 其他基因的变异策略

        return population
```

### 2. 模拟退火 (Simulated Annealing)

```python
import math
import random

class SimulatedAnnealingScheduler:
    def __init__(self, initial_temp=1000.0, cooling_rate=0.95, min_temp=1.0):
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp

    def optimize(self, search_space, max_iterations=1000):
        """模拟退火主流程"""
        # 1. 初始解
        current_solution = self.random_solution(search_space)
        current_cost = self.evaluate_cost(current_solution)

        best_solution = current_solution
        best_cost = current_cost

        temperature = self.initial_temp
        iteration = 0

        while temperature > self.min_temp and iteration < max_iterations:
            iteration += 1

            # 2. 生成邻域解
            neighbor = self.generate_neighbor(current_solution)
            neighbor_cost = self.evaluate_cost(neighbor)

            # 3. 计算能量差
            delta_cost = neighbor_cost - current_cost

            # 4. 接受判断
            if delta_cost < 0:  # 邻域解更好，直接接受
                current_solution = neighbor
                current_cost = neighbor_cost

                if current_cost < best_cost:
                    best_solution = current_solution
                    best_cost = current_cost

            else:  # 邻域解更差，按概率接受
                probability = math.exp(-delta_cost / temperature)
                if random.random() < probability:
                    current_solution = neighbor
                    current_cost = neighbor_cost

            # 5. 降温
            temperature *= self.cooling_rate

            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Best cost = {best_cost}, Temp = {temperature}")

        return best_solution

    def generate_neighbor(self, solution):
        """生成当前解的邻域解"""
        neighbor = solution.copy()

        # 随机选择一个变换操作
        transformations = [
            self.mutate_tile_size,
            self.mutate_parallel_dim,
            self.mutate_vectorize_dim,
            self.mutate_loop_order
        ]

        transform = random.choice(transformations)
        return transform(neighbor)

    def mutate_tile_size(self, solution):
        """变异分块大小"""
        current_tile = solution['tile_size']
        new_tile = random.choice([2, 4, 8, 16, 32, 64])
        solution['tile_size'] = new_tile
        return solution
```

### 3. 强化学习方法

```python
import numpy as np
from collections import defaultdict

class RLBasedScheduler:
    def __init__(self, state_dim=10, action_dim=20, learning_rate=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate

        # Q表：状态-动作值函数
        self.q_table = defaultdict(lambda: np.zeros(action_dim))

        # ε-贪婪策略参数
        self.epsilon = 0.1  # 探索概率
        self.gamma = 0.9    # 折扣因子

    def state_from_schedule(self, schedule):
        """将调度转换为状态表示"""
        state = np.zeros(self.state_dim)

        # 编码调度的关键特征
        state[0] = schedule['tile_size'] / 64.0        # 归一化的分块大小
        state[1] = len(schedule['parallel_dims']) / 3.0  # 并行维度数量
        state[2] = schedule['vectorize_dim'] / 2.0     # 向量化维度
        state[3] = schedule['unroll_factor'] / 8.0      # 展开因子
        # ... 更多特征

        return tuple(state)  # 转为tuple以便作为字典key

    def choose_action(self, state):
        """ε-贪婪策略选择动作"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)  # 探索
        else:
            return np.argmax(self.q_table[state])          # 利用

    def action_to_schedule_change(self, action, current_schedule):
        """将动作转换为调度变更"""
        new_schedule = current_schedule.copy()

        if action < 5:  # 改变分块大小
            tile_sizes = [2, 4, 8, 16, 32]
            new_schedule['tile_size'] = tile_sizes[action]
        elif action < 10:  # 改变并行维度
            parallel_configs = [
                [0], [1], [2], [0, 1], [0, 2], [1, 2]
            ]
            new_schedule['parallel_dims'] = parallel_configs[action - 5]
        # ... 其他动作

        return new_schedule

    def learn(self, num_episodes=1000):
        """强化学习主循环"""
        best_schedule = None
        best_performance = float('inf')

        for episode in range(num_episodes):
            # 初始状态
            current_schedule = self.random_schedule()
            current_state = self.state_from_schedule(current_schedule)

            done = False
            step = 0

            while not done and step < 20:  # 最多20步
                # 选择动作
                action = self.choose_action(current_state)

                # 执行动作，获得新状态
                new_schedule = self.action_to_schedule_change(action, current_schedule)
                new_state = self.state_from_schedule(new_schedule)

                # 计算奖励
                performance = self.evaluate_schedule(new_schedule)
                reward = self.calculate_reward(current_schedule, new_schedule, performance)

                # 更新Q值
                old_value = self.q_table[current_state][action]
                next_max = np.max(self.q_table[new_state])
                new_value = old_value + self.learning_rate * (reward + self.gamma * next_max - old_value)
                self.q_table[current_state][action] = new_value

                # 更新状态
                current_schedule = new_schedule
                current_state = new_state

                # 记录最佳结果
                if performance < best_performance:
                    best_performance = performance
                    best_schedule = current_schedule

                step += 1

                # 检查是否收敛
                if step > 5 and performance < 0.001:  # 性能已经很好
                    done = True

            # 衰减探索概率
            self.epsilon *= 0.995

            if episode % 100 == 0:
                print(f"Episode {episode}: Best performance = {best_performance:.6f}")

        return best_schedule
```

## 🔧 实际实现细节

### 1. 搜索空间的智能构建

TVM不会盲目尝试所有组合，而是使用启发式方法：

```python
class SmartSearchSpace:
    def __init__(self, target_device):
        self.target = target_device
        self.device_specific_rules = self.get_device_rules()

    def get_device_rules(self):
        """根据设备特性定制搜索规则"""
        if self.target == "cuda":
            return {
                'tile_sizes': [8, 16, 32],      # GPU适合较大的tile
                'vectorize': False,             # GPU不需要向量化
                'parallel': True,               # GPU天然并行
                'shared_memory': True           # GPU有共享内存
            }
        elif self.target == "llvm":
            return {
                'tile_sizes': [4, 8, 16],      # CPU适合中等tile
                'vectorize': [4, 8, 16],       # CPU支持SIMD
                'parallel': True,               # CPU多核并行
                'shared_memory': False          # CPU没有GPU式共享内存
            }

    def generate_smart_space(self, computation):
        """智能生成搜索空间"""
        space = []

        # 基于计算模式推荐方案
        if self.is_matrix_multiply(computation):
            space.extend(self.generate_matmul_space())
        elif self.is_convolution(computation):
            space.extend(self.generate_conv_space())
        else:
            space.extend(self.generate_general_space())

        return space

    def is_matrix_multiply(self, computation):
        """检测是否是矩阵乘法"""
        # 通过分析计算图模式来判断
        return "matmul" in computation.name.lower()
```

### 2. 早停和剪枝策略

```python
class EarlyStopping:
    def __init__(self, patience=20, min_improvement=0.01):
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_score = float('inf')
        self.wait = 0

    def should_stop(self, current_score):
        """判断是否应该早停"""
        if current_score < self.best_score - self.min_improvement:
            self.best_score = current_score
            self.wait = 0
        else:
            self.wait += 1

        return self.wait >= self.patience

class SearchSpacePruning:
    def __init__(self):
        self.performance_cache = {}
        self.rule_based_filters = [
            self.filter_impossible_combinations,
            self.filter_known_bad_patterns,
            self.filter_redundant_options
        ]

    def prune_space(self, search_space):
        """剪枝搜索空间"""
        pruned_space = search_space.copy()

        for filter_func in self.rule_based_filters:
            pruned_space = filter_func(pruned_space)

        # 基于历史性能剪枝
        pruned_space = self.history_based_pruning(pruned_space)

        return pruned_space

    def history_based_pruning(self, search_space):
        """基于历史性能数据剪枝"""
        filtered_space = []

        for candidate in search_space:
            # 检查是否有相似的已知性能数据
            similar_candidates = self.find_similar_candidates(candidate)

            if similar_candidates:
                avg_performance = np.mean([self.performance_cache.get(c, float('inf'))
                                         for c in similar_candidates])

                # 如果相似候选性能很差，跳过
                if avg_performance > self.threshold:
                    continue

            filtered_space.append(candidate)

        return filtered_space
```

### 3. 性能预测模型

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

class PerformancePredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False

    def extract_features(self, schedule):
        """从调度中提取特征"""
        features = []

        # 结构特征
        features.extend([
            schedule['tile_size'],
            len(schedule['parallel_dims']),
            schedule['vectorize_dim'] if 'vectorize_dim' in schedule else 0,
            schedule['unroll_factor'],
        ])

        # 循环嵌套特征
        features.extend([
            schedule.get('loop_nest_depth', 1),
            schedule.get('total_iterations', 1),
        ])

        # 内存访问特征
        features.extend([
            schedule.get('memory_footprint', 0),
            schedule.get('cache_efficiency', 0),
        ])

        return features

    def train(self, schedules, performance_data):
        """训练性能预测模型"""
        X = np.array([self.extract_features(s) for s in schedules])
        y = np.array(performance_data)

        # 特征标准化
        X_scaled = self.scaler.fit_transform(X)

        # 训练模型
        self.model.fit(X_scaled, y)
        self.is_trained = True

    def predict(self, schedule):
        """预测调度性能"""
        if not self.is_trained:
            return None

        features = self.extract_features(schedule)
        features_scaled = self.scaler.transform([features])

        return self.model.predict(features_scaled)[0]
```

## 🎮 完整实例：矩阵乘法自动调优

```python
import tvm
from tvm import te, auto_scheduler
import numpy as np

class MatrixMultiplicationTuner:
    def __init__(self, M=1024, N=1024, K=1024):
        self.M, self.N, self.K = M, N, K

    def create_computation(self):
        """定义矩阵乘法计算"""
        A = te.placeholder((self.M, self.K), name='A')
        B = te.placeholder((self.K, self.N), name='B')

        k = te.reduce_axis((0, self.K), name='k')
        C = te.compute((self.M, self.N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name='C')

        return A, B, C

    def auto_tune(self, num_trials=1000):
        """执行自动调优"""
        # 创建计算
        A, B, C = self.create_computation()

        # 创建搜索任务
        task = auto_scheduler.SearchTask(
            func_name="matmul",
            args=[A, B, C],
            target="llvm"
        )

        print(f"搜索空间大小: {len(auto_scheduler.measure._get_task(task))}")

        # 配置调优选项
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=num_trials,           # 总试验次数
            num_measure_trials_per_iter=64,         # 每次迭代的试验次数
            early_stopping=50,                      # 早停轮数

            # 构建器配置
            builder=auto_scheduler.LocalBuilder(),

            # 运行器配置
            runner=auto_scheduler.LocalRunner(
                repeat=3,                           # 每个方案运行3次取平均
                min_repeat_ms=100,                # 最小运行时间
                enable_cpu_cache_flush=True       # 清除CPU缓存
            ),

            # 测量回调
            measure_callbacks=[
                auto_scheduler.RecordToFile("matmul_tuning.json")
            ]
        )

        # 执行自动调优
        print("开始自动调优...")
        sch, args = auto_scheduler.auto_task_tune(task, tune_option)

        print("调优完成！")
        return sch, args

    def evaluate_performance(self, lib, num_trials=100):
        """评估编译后模块的性能"""
        # 准备测试数据
        dev = tvm.cpu(0)
        a = tvm.nd.array(np.random.randn(self.M, self.K).astype("float32"), dev)
        b = tvm.nd.array(np.random.randn(self.K, self.N).astype("float32"), dev)
        c = tvm.nd.array(np.zeros((self.M, self.N), dtype="float32"), dev)

        # 创建图执行器
        module = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
        module.set_input("A", a)
        module.set_input("B", b)
        module.set_input("C", c)

        # 预热
        for _ in range(10):
            module.run()

        # 性能测试
        import time
        times = []

        for _ in range(num_trials):
            start_time = time.perf_counter()
            module.run()
            end_time = time.perf_counter()
            times.append(end_time - start_time)

        avg_time = np.mean(times)
        gflops = (2 * self.M * self.N * self.K) / (avg_time * 1e9)

        return avg_time, gflops

    def compare_with_baseline(self, tuned_lib):
        """与基准实现对比"""
        print("=== 性能对比 ===")

        # 调优后性能
        tuned_time, tuned_gflops = self.evaluate_performance(tuned_lib)

        # 基准性能（简单实现）
        A, B, C = self.create_computation()
        s = te.create_schedule(C.op)
        baseline_lib = tvm.build(s, [A, B, C], target="llvm")

        baseline_time, baseline_gflops = self.evaluate_performance(baseline_lib)

        print(f"基准实现:   {baseline_time:.6f}s, {baseline_gflops:.2f} GFLOPS")
        print(f"调优实现:   {tuned_time:.6f}s, {tuned_gflops:.2f} GFLOPS")
        print(f"性能提升:   {baseline_time/tuned_time:.2f}x")
        print(f"GFLOPS提升: {tuned_gflops/baseline_gflops:.2f}x")

# 执行调优
if __name__ == "__main__":
    tuner = MatrixMultiplicationTuner(M=512, N=512, K=512)

    # 自动调优
    schedule, args = tuner.auto_tune(num_trials=200)

    # 编译调优后的模块
    tuned_lib = tvm.build(schedule, args, target="llvm")

    # 性能对比
    tuner.compare_with_baseline(tuned_lib)
```

## 📊 调优结果分析与可视化

```python
import matplotlib.pyplot as plt
import pandas as pd
import json

class TuningAnalyzer:
    def __init__(self, log_file="matmul_tuning.json"):
        self.log_file = log_file
        self.data = self.load_tuning_data()

    def load_tuning_data(self):
        """加载调优日志数据"""
        with open(self.log_file, 'r') as f:
            records = []
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    records.append(record)
            return records

    def plot_convergence(self):
        """绘制收敛曲线"""
        trials = []
        costs = []

        for record in self.data:
            if record['result'][0]['costs'] != []:
                trials.append(record['config_index'])
                costs.append(min(record['result'][0]['costs']))

        # 计算累积最佳性能
        best_so_far = []
        current_best = float('inf')
        for cost in costs:
            if cost < current_best:
                current_best = cost
            best_so_far.append(current_best)

        plt.figure(figsize=(12, 8))

        # 原始性能散点
        plt.subplot(2, 1, 1)
        plt.scatter(trials, costs, alpha=0.6, s=10)
        plt.xlabel('Trial Number')
        plt.ylabel('Execution Time (ms)')
        plt.title('Performance of Each Trial')
        plt.grid(True, alpha=0.3)

        # 收敛曲线
        plt.subplot(2, 1, 2)
        plt.plot(trials[:len(best_so_far)], best_so_far, 'b-', linewidth=2)
        plt.xlabel('Trial Number')
        plt.ylabel('Best Time So Far (ms)')
        plt.title('Convergence Curve')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def analyze_parameter_importance(self):
        """分析不同参数的重要性"""
        df = pd.DataFrame([
            {
                'tile_size': record['config']['tile_size'],
                'parallel_dim': record['config']['parallel_dim'],
                'performance': min(record['result'][0]['costs'])
            }
            for record in self.data
            if record['result'][0]['costs'] != []
        ])

        # 参数相关性分析
        correlations = df.corr()['performance']
        print("参数与性能的相关性:")
        print(correlations)

        # 参数分布分析
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        df.boxplot(column='performance', by='tile_size')
        plt.title('Performance by Tile Size')

        plt.subplot(1, 3, 2)
        df.boxplot(column='performance', by='parallel_dim')
        plt.title('Performance by Parallel Dimension')

        plt.tight_layout()
        plt.show()

    def generate_report(self):
        """生成调优报告"""
        if not self.data:
            print("没有调优数据")
            return

        successful_trials = [r for r in self.data if r['result'][0]['costs'] != []]

        if not successful_trials:
            print("没有成功的调优试验")
            return

        costs = [min(r['result'][0]['costs']) for r in successful_trials]

        print("=== TVM自动调优报告 ===")
        print(f"总试验次数: {len(self.data)}")
        print(f"成功试验: {len(successful_trials)}")
        print(f"成功率: {len(successful_trials)/len(self.data)*100:.1f}%")
        print(f"最佳性能: {min(costs):.6f}ms")
        print(f"平均性能: {np.mean(costs):.6f}ms")
        print(f"性能标准差: {np.std(costs):.6f}ms")

        # 显示最佳配置
        best_trial = successful_trials[np.argmin(costs)]
        print("\n最佳配置:")
        for key, value in best_trial['config'].items():
            print(f"  {key}: {value}")
```

## 🔮 未来发展方向

### 1. 机器学习增强
```python
# 使用深度学习预测性能
import torch
import torch.nn as nn

class DeepPerformancePredictor(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.network(x)
```

### 2. 多目标优化
```python
# 同时优化性能、功耗、内存使用
class MultiObjectiveOptimizer:
    def __init__(self, weights={'performance': 0.6, 'power': 0.3, 'memory': 0.1}):
        self.weights = weights

    def evaluate_multi_objective(self, schedule):
        perf = self.measure_performance(schedule)
        power = self.measure_power_consumption(schedule)
        memory = self.measure_memory_usage(schedule)

        # 加权综合评分
        score = (self.weights['performance'] * (1/perf) +
                self.weights['power'] * (1/power) +
                self.weights['memory'] * (1/memory))

        return score, {'performance': perf, 'power': power, 'memory': memory}
```

### 3. 迁移学习
```python
# 将一个任务的调优经验迁移到其他任务
class TransferLearningTuner:
    def __init__(self):
        self.source_tasks = []
        self.transfer_model = None

    def learn_from_source(self, source_tasks):
        """从源任务学习调优知识"""
        self.source_tasks = source_tasks
        # 训练迁移模型...

    def tune_target_task(self, target_task, num_trials=100):
        """使用迁移学习调优目标任务"""
        # 从源任务知识初始化搜索...
        pass
```

## 🎉 总结

TVM的自动搜索最佳调度系统是一个复杂的智能系统，它：

### 核心创新
1. **搜索空间智能构建**：基于硬件特性和计算模式
2. **多算法融合**：遗传、模拟退火、强化学习等方法结合
3. **性能预测**：机器学习模型预测性能，减少实际测量次数
4. **自适应优化**：根据中间结果调整搜索策略

### 实现特点
1. **启发式剪枝**：避免盲目搜索，提高效率
2. **早停机制**：及时发现收敛，节省时间
3. **并行执行**：多线程构建和测试
4. **结果缓存**：避免重复计算

### 应用价值
1. **自动化**：减少手动调优工作量
2. **高效性**：往往能找到人工难以发现的优化方案
3. **通用性**：适用于各种计算模式
4. **可扩展性**：易于添加新的优化策略

这个系统代表了AI编译器发展的前沿方向，通过将传统编译器技术与现代机器学习方法结合，实现了编译优化的智能化和自动化。

---

**下一步**：可以尝试将自己的模型用TVM自动调优，体验这个强大系统的魔力！🚀