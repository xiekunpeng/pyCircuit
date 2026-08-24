# PyCircuit V6 语言规范（Specification）

**版本：6.0**

**适用代码基线**：`pyCircuit` 主仓库 `main` 分支（含类型化 `Data` 信号体系、`Wire[Vector]`、ASL 对齐的类型级语法糖、Sidecar 测试调度、层次化 MLIR 发射）。

> **包名**：Python 包为 **`pycircuit`**（全小写）。不要使用 `pyCircuit` 作为 import 目标。
>
> **与 V5 的关系**：V6 完整继承 V5 的周期感知（Cycle-Aware）信号模型与层次化组合规范，并将以下能力纳入正式语言规范：
> 1. **类型化数据体系**（`Data` 层级：`Bits` / `Vector` / `Clock` / `Reset`，由 `Wire[DT]` 统一承载；逐 lane 运算、降维归约、广播）
> 2. **类型级语法糖**（位域视图、位掩码模式匹配、类型安全枚举、受检转换、缩放切片、显式签名比较——与 ARM ASL 数据类型对齐，见 §3.8–§3.11）
> 3. **Sidecar 测试调度**（测试激励与检查外置为二进制容器）
> 4. **双编译路径**（`compile_cycle_aware` eager/JIT 与 `@module` JIT）
> 5. **存储 / FIFO / CDC 原语**

---

## 目录

1. [概述与设计哲学](#1-概述与设计哲学)
2. [信号类型纪律（Non-Negotiable）](#2-信号类型纪律non-negotiable)
3. [核心类型与数据类型体系](#3-核心类型与数据类型体系)
   - 3.8 [位域视图（`BitfieldSpec`）](#38-位域视图bitfieldspec)
   - 3.9 [位掩码模式匹配（`matches` / `in_`）](#39-位掩码模式匹配matches--in_)
   - 3.10 [类型安全枚举（`PycEnum`）](#310-类型安全枚举pycenum)
   - 3.11 [受检转换、缩放切片与显式签名比较](#311-受检转换缩放切片与显式签名比较)
4. [Forward Signal 与寄存器推导](#4-forward-signal-与寄存器推导)
5. [全局函数](#5-全局函数)
6. [周期管理与自动周期平衡](#6-周期管理与自动周期平衡)
7. [模块签名与层次化组合](#7-模块签名与层次化组合)
8. [存储 / FIFO / CDC 原语](#8-存储--fifo--cdc-原语)
9. [仿真与测试（Tb / CycleAwareTb / Sidecar）](#9-仿真与测试tb--cycleawaretb--sidecar)
10. [编译入口](#10-编译入口)
11. [MLIR 映射参考](#11-mlir-映射参考)
12. [Tier 分层标注（3D 堆叠扩展，Proposed）](#12-tier-分层标注3d-堆叠扩展proposed)
13. [附录：API 参考表](#13-附录api-参考表)

---

## 1. 概述与设计哲学

PyCircuit 是嵌入在 Python 中的硬件描述语言（HDL）。设计者用普通 Python 函数描述电路，前端将其编译为 MLIR（`pyc` 方言），后端 `pycc` 再生成可综合 Verilog 与周期精确的 C++ 仿真模型。

V6 的核心概念：

- **周期感知信号（`CycleAwareSignal`，简称 CAS）**：设计中流动的唯一信号类型。每个信号携带「逻辑周期」标签，编译器根据周期差**自动插入 DFF** 完成流水线对齐。
- **前向声明寄存器（`domain.signal()` + `<<=`）**：先声明后赋值，读写周期差决定寄存器级数——寄存器是**推导**出来的，不是手写的。
- **层次化组合（`domain.call()`）**：Python 函数调用链即硬件模块层次；每个模块既能独立编译，又能被父模块组合。
- **类型化数据体系（`Data` / `Wire[DT]`）**：`Bits` / `Vector` / `Clock` / `Reset` 四种 `Data` 子类由统一的 `Wire[DT]` 句柄承载；向量 `vector<D0x...xiW>` 是一等公民，SIMD 运算逐 lane 并行，可保留到 Verilog（packed bus）或按需标量化。
- **周期精确仿真**：同一 MLIR 同时生成 Verilog 和 C++ 仿真器，测试台（`CycleAwareTb`）用与设计对称的 `next()` 模型编写。

### 标准导入

```python
from pycircuit import (
    # ── V6 核心（周期感知模型）──
    CycleAwareCircuit,    # 顶层电路
    CycleAwareDomain,     # 周期感知时钟域
    CycleAwareSignal,     # 周期感知信号（唯一信号类型）
    ForwardSignal,        # domain.signal() 的返回类型
    cas,                  # Wire → CycleAwareSignal
    compile_cycle_aware,  # V6 编译入口
    mux,                  # 多路选择器
    submodule_input,      # 双模输入辅助
    wire_of,              # 边界提取 Wire（仅用于 m.output()）
    # ── 测试 ──
    Tb, CycleAwareTb, testbench,
    # ── 位操作辅助 ──
    cat, zext, sext, trunc,
    # ── 类型级语法糖（§3.8–§3.11）──
    BitfieldSpec,          # 位域布局（view / update / bind）
    BitfieldSignal,        # 绑定了位域布局的信号
    PycEnum, auto,         # 类型安全枚举（class 形式）
    enumeration,           # 类型安全枚举（ASL 式一行声明）
)
```

---

## 2. 信号类型纪律（Non-Negotiable）

PyCircuit V6 强制单一信号类型。以下规则**不可违反**：

| 规则 | 说明 |
|------|------|
| **所有标量信号都是 `CycleAwareSignal`** | 设计中流动的每个标量值必须是 `CycleAwareSignal`（或 `ForwardSignal`，其读侧委托给 CAS）。 |
| **`domain.state()` 不存在** | 创建寄存器的唯一方式是 `domain.signal()` + `<<=` / `.assign()`。 |
| **`.wire` / `.w` 不可访问** | 所有算术、比较、mux、切片直接在 CAS 上进行。设计代码中读取 `.wire` 是错误。 |
| **`wire_of()` 是唯一的 Wire 提取方式** | 仅在 `m.output()` 边界调用。 |
| **`cas()` 包装裸 Wire** | `m.input()` / `m.const()` 返回裸 `Wire`，必须用 `cas(domain, w, cycle=0)` 或 `submodule_input()` 包装后才能参与 CAS 表达式。 |
| **输出 dict 存 CAS** | 子模块返回的 dict 值必须是 `CycleAwareSignal`（保留 cycle 来源信息）。 |
| **向量走 `Wire[Vector]`** | 多 lane 同构数据统一用 `m.input(..., shape=[N])` 声明为向量 `Wire[Vector[...]]`，不要手工拆 lane 拼标量。 |

```python
# ✅ 正确
pc = domain.signal(width=64, reset_value=0, name="pc")
enable = cas(domain, m.input("en", width=1), cycle=0)
result = pc + enable            # CAS + CAS → CAS（自动周期对齐）
m.output("pc", wire_of(pc))     # wire_of() 仅在边界

# ❌ 禁止
st = domain.state(...)          # 不存在
raw = pc.wire                   # 不可访问
outs["result"] = wire_of(r)     # dict 里必须存 CAS
x = a.wire + b.wire             # 直接在 CAS 上运算
```

---

## 3. 核心类型与数据类型体系

PyCircuit 的所有硬件对象都建立在两个相互正交的概念之上：

- **数据类型 `Data`**：描述信号承载的「值的形状」——标量位宽、向量维度、时钟/复位语义；
- **信号句柄 `Wire[DT]`**：描述信号在电路图里的「身份」——一个端口、一条连线、一个寄存器输出。

前者决定运算的合法性与 MLIR 类型；后者承载重载运算符与周期感知包装。两者通过泛型参数 `DT`（绑定到 `Data`）关联：标量是 `Wire[Bits]`，向量是 `Wire[Vector[...]]`，它们共享同一套运算符接口。

### 3.1 Data 类型层级

`pycircuit.data` 定义了一个冻结的（`frozen` dataclass）类型层级，`str(Data)` 直接给出 MLIR 类型字面量：

```
Data (ABC)
├── Bits(bitwidth: int)          → iN        （标量位向量）
├── Vector(length: int, elem: DT) → vector<Nx...xiW>  （可嵌套，多维）
├── Clock                         → !pyc.clock
└── Reset                         → !pyc.reset
```

| 子类 | MLIR 字面量 | `.width` 语义 | 说明 |
|------|------------|--------------|------|
| `Bits(N)` | `iN` | N | N 位标量；所有算术/逻辑/比较运算的操作数 |
| `Vector(N, elem)` | `vector<NxiW>` | 叶元素宽度 | 一维向量；`elem` 可再为 `Vector`，得到多维 `vector<NxMxiW>` |
| `Clock` | `!pyc.clock` | 1 | 时钟信号，仅作 `pyc.reg` 的时钟输入 |
| `Reset` | `!pyc.reset` | 1 | 复位信号，仅作 `pyc.reg` 的复位输入 |

`Vector` 的两个辅助方法：

| 方法 | 说明 |
|------|------|
| `.shape()` | 所有维度扁平化，外→内。`Vector(4, Vector(3, Bits(8)))` → `[4, 3]` |
| `.datatype()` | 最内层非 `Vector` 叶元素。`Vector(4, Vector(3, Bits(8)))` → `Bits(8)` |
| `Vector.from_shape([N, M], leaf)` | 反向构造：把 `leaf` 由外到内包成嵌套 `Vector` |

> `Data` 实例可比较、可哈希（以 MLIR 字面量为键），并可用 `Data.from_str("vector<4xi8>")` 从 MLIR 文本反序列化。

### 3.2 Wire[DT] —— 信号句柄

`Wire` 是端口语义的统一句柄，用 `DT`（绑定 `Data`）泛型参数化承载上述类型：

```python
a = m.input("a", width=8)                  # Wire[Bits[8]]     → i8
v = m.input("v", width=8, shape=[4])       # Wire[Vector[Bits[8]]]  → vector<4xi8>
mat = m.input("m", width=8, shape=[4,16])  # Wire[Vector[...]]      → vector<4x16xi8>
```

`Wire[Bits]` 与 `Wire[Vector[...]]` **共享同一套重载运算符**——区别仅在结果类型：

```python
r   = a + b              # Wire[Bits]  + Wire[Bits]   → Wire[Bits]
s   = v + w              # Wire[Vector] + Wire[Vector] → Wire[Vector]（逐 lane）
eqv = v == w             # → Wire[Vector[Bits[1]]]（逐 lane 比较）
bc  = v + a              # 标量自动广播到每个 lane
sel = mux(cond_vec, v, w)  # 逐 lane mux
```

完整运算符集（标量与向量语义一致，向量逐 lane 并行，标量操作数自动广播）：

```python
r = a + b;  r = a - b;  r = a * b          # 算术
r = a & b;  r = a | b;  r = a ^ b;  r = ~a  # 位运算
r = a == b; r = a != b                      # 比较（也可 a.eq(b) / a.ne(b)）
r = a < b;  r = a > b;  r = a <= b; r = a >= b
r = a << k; r = a >> k                      # 移位
low = data[0:8];  bit5 = data[5]            # 切片 / 索引（向量：取 lane）
```

通用方法（标量与向量 `Wire` 都有）：

| 方法 | 说明 |
|------|------|
| `select(t, f)` | 条件选择（等价 `mux(self, t, f)`） |
| `trunc(width=)` / `zext(width=)` / `sext(width=)` | 宽度变换 |
| `slice(*, lsb, width)` | 位片段（等价 `x[lsb : lsb+width]`） |
| `lane(idx, *, width)` / `x[idx, width]` | 缩放切片（按元素取，见 §3.11） |
| `named(name)` | 调试名称 |
| `as_signed()` / `as_unsigned()` | 符号标记（影响比较与右移语义） |
| `ult / ugt / ule / uge` | 显式无符号比较 |
| `slt / sgt / sle / sge` | 显式有符号比较（见 §3.11 的显式签名约定） |
| `matches / in_ / not_in_` | 位掩码模式匹配（见 §3.9） |
| `as_ / assert_fits / assert_range / assert_in` | 受检转换（见 §3.11） |

> `Wire` **禁止**作为 Python `bool` 使用（`if sig:` 报错）——硬件条件必须用 `mux` / `select` 表达。

### 3.3 向量专属方法（降维 / 广播 / 拼接）

只有 `Wire[Vector[...]]` 才有的方法（作用于 `Vector` 维度结构）：

| 方法 | 说明 | MLIR |
|------|------|------|
| `v.reduce_or(*, dim=None, mode="chain")` | 按位或归约；`dim=None` 全维归约成标量，`dim=int` 只归约指定维 | `pyc.v_or_reduce` |
| `v.reduce_and(*, dim=None, mode="chain")` | 按位与归约；同上 | `pyc.v_and_reduce` |
| `v.reduce_sum(*, dim=None, mode="chain")` | 求和归约；**保持叶元素宽度，溢出回绕**（不会自动扩展结果宽度） | `pyc.v_add_reduce` |
| `v.broadcast(*, size, dim)` | 沿新维度重复 | `pyc.v_broadcast_dim` |
| `priority_mux(sels, vals, *, mode="chain", default=None)` | 以 `sels`（i1 向量）为选择器在 `vals` 中选 lane；**最小索引优先**；模块级函数或 CAS 实例方法 `sels.priority_mux(vals, ...)`；`default=None` 时所有 selector 为 0 的回退值是 `vals` 的最后一个元素 | 组合展开 |
| `m.cat(*lanes)` | 把多个 lane 拼成 packed 标量位向量（MSB-first） | `pyc.concat` |
| `v[i]` | 取 lane | `pyc.v_get` |

`mode` 取值：`"chain"`（默认，逻辑深度 = lanes−1）或 `"tree"`（深度 = ⌈log₂(lanes)⌉）。深度敏感的宽归约应显式选 `"tree"`。

`dim` 指定多维向量的归约维度（如 `reduce_or(dim=0)` 把 `vector<4x16xi1>` 归约为 `vector<16xi1>`）；省略 `dim` 表示跨所有维归约成标量。

**典型模式：旁路匹配矩阵**

```python
# 每个源操作数 tag 与所有写回 tag 广播比较，再按维度归约
src_b  = src_tag_vec.broadcast(dim=1, size=n_wb)       # [n_src × n_wb]
wb_b   = wb_tag_vec.broadcast(dim=0, size=n_src)       # [n_src × n_wb]
hit    = (src_b == wb_b) & wb_valid_b                  # 匹配矩阵
any_hit = hit.reduce_or(dim=1)                            # 每 src 是否命中
data0   = priority_mux(hit[0], wb_data_vec, default=zero_data)   # src 0 的旁路数据
```

### 3.4 CycleAwareCircuit

顶层电路对象，`Circuit` 的子类。

```python
m = CycleAwareCircuit("my_circuit")
```

| 方法 | 说明 |
|------|------|
| `create_domain(name, *, frequency_desc="", reset_active_high=False)` | 创建 `CycleAwareDomain` |
| `input(name, *, width, signed=False, shape=None, fields=None, enum=None)` | 输入端口。`shape=None` 返回标量 `Wire[Bits]`；`shape=[N]` / `shape=[N,M]` 返回 `Wire[Vector[...]]`（需 `cas()` 包装后参与运算）。`fields=` 返回 `BitfieldSignal`（§3.8），`enum=` 返回 `EnumSignal`（§3.10）；三者互斥 |
| `output(name, value)` | 注册输出端口（标量传 `wire_of(sig)`；向量 `Wire` 可直接传） |
| `const(value, *, width)` | 常量 `Wire`（用 `cas()` 包装后参与 CAS 表达式） |
| `vec(*elems)` / `cat(parts)` | 从已有元素构造 `Wire[Vector]`（可变参数） / 位拼接 |
| `emit_mlir()` | 生成 MLIR 文本。层次化编译时输出含所有子模块的多模块 `Design` |

### 3.5 CycleAwareDomain

管理一个时钟域的逻辑周期状态。

```python
domain = m.create_domain("clk")
```

| 方法 | 说明 |
|------|------|
| `signal(*, width, reset_value=0, name="", shape=None, fields=None, enum=None)` | **前向声明寄存器**——创建状态的唯一方式。`shape=` 时创建向量状态；返回 `ForwardSignal`。`fields=` 返回 `BitfieldSignal`（§3.8），`enum=` 返回 `EnumSignal`（§3.10）；三者互斥 |
| `cycle(sig, reset_value=None, name="")` | 对信号插入单级 DFF，返回延后一拍的 CAS |
| `next()` / `prev()` | 推进 / 回退当前逻辑周期 |
| `push()` / `pop()` | 周期计数器压栈 / 出栈（必须配对） |
| `call(fn, *, inputs=None, **kwargs)` | 调用子模块并自动 push/pop 隔离周期。扁平模式内联；层次化模式发射 `pyc.instance` |
| `delay_to(w, *, from_cycle, to_cycle, width)` | 显式打拍对齐（自动平衡的底层机制） |
| `create_signal(name, *, width, shape=None, signed=False)` | 创建输入端口（裸 `Wire`，标量或向量） |
| `create_const(value, *, width, name="", signed=False)` | 常量 `Wire` |
| `create_reset()` | 复位信号（有效高视图，i1 `Wire`） |
| `cycle_index` | 属性：当前逻辑周期索引 |

多时钟域：

```python
cpu_clk = m.create_domain("CPU_CLK", frequency_desc="100MHz")
rtc_clk = m.create_domain("RTC_CLK", frequency_desc="1Hz")
```

跨时钟域信号**必须**经显式 CDC 原语（`cdc_sync` / `async_fifo`）传递；后端 `pyc-check-clock-domains` 检查违例并报错。

### 3.6 CycleAwareSignal

唯一的标量信号类型。包含底层线网（内部管理）、`cycle`、`domain`。

**属性**：`cycle`、`domain`、`name`、`signed`。

**运算符**（所有输入输出均为 CAS，自动周期对齐）：

```python
r = a + b;  r = a - b;  r = a * b          # 算术
r = a & b;  r = a | b;  r = a ^ b;  r = ~a  # 位运算
r = a == b; r = a != b                      # 比较（也可 a.eq(b) / a.ne(b)）
r = a < b;  r = a > b;  r = a <= b; r = a >= b
r = a << k; r = a >> k                      # 移位
low = data[0:8];  bit5 = data[5]            # 切片 / 索引
```

**方法**：

| 方法 | 说明 |
|------|------|
| `select(t, f)` | 条件选择（等价 `mux(self, t, f)`） |
| `trunc(w)` / `zext(w)` / `sext(w)` | 宽度变换 |
| `slice(high, low)` / `slice(*, lsb, width)` | 位片段（两种等价写法，闭区间 / 起点+宽度） |
| `lane(idx, *, width)` / `x[idx, width]` | 缩放切片（见 §3.11） |
| `named(name)` | 调试名称 |
| `as_signed()` / `as_unsigned()` | 符号标记（影响比较与右移语义） |
| `ult / ugt / ule / uge`、`slt / sgt / sle / sge` | 显式签名比较（全套，见 §3.11） |
| `matches / in_ / not_in_` | 位掩码模式匹配（见 §3.9） |
| `as_ / assert_fits / assert_range / assert_in` | 受检转换（见 §3.11） |

> CAS **禁止**作为 Python `bool` 使用（`if sig:` 报错）——硬件条件必须用 `mux` / `select` 表达。
>
> 上述 §3.8–§3.11 的语法糖在 `Wire` 与 CAS 上**语义一致**；CAS 版本保留 `cycle` 标签。

### 3.7 ForwardSignal

`domain.signal()` 的返回类型。读侧行为与 CAS 完全一致；额外提供写侧接口：

```python
sig <<= expr                     # 无条件赋值（连接 D 端）
sig.assign(expr, when=cond)      # 条件赋值（寄存器使能）
```

### 3.8 位域视图（`BitfieldSpec`）

给**单个信号**的位区间起名字，取代解码/系统寄存器代码里的 `instr[25:21]` 魔法数字。字段范围写成 `(msb, lsb)` **闭区间**（与架构手册、Verilog 一致），**允许字段重叠**——同一寄存器的多种视图是合法的。

```python
from pycircuit import BitfieldSpec

INSTR = BitfieldSpec(width=32, fields={
    "opcode": (31, 26), "rd": (25, 21), "rs": (20, 16), "imm16": (15, 0),
    "imm26":  (25, 0),                 # 与 rd/rs/imm16 重叠 —— 合法（不同视图）
})
```

**用法 A：临时视图**（对已有信号即用即取，不改变原信号）

```python
f  = INSTR.view(instr)                 # 或简写 INSTR(instr)
op = f["opcode"]                       # ≡ instr[26:32]
op = f.opcode                          # 属性形式，等价
hi = f["opcode", "rd"]                 # 多字段拼接读（MSB-first）≡ cat(...)
wr = INSTR.update(instr, rd=new_rd)    # 读-改-写 ≡ cat(instr[26:32], new_rd, instr[0:21])
```

**用法 B：声明即绑定**（声明处指示布局，之后直接按名访问）

```python
instr = m.input("instr", width=32, fields={          # 传字典须给 width
    "opcode": (31, 26), "rd": (25, 21), "imm16": (15, 0),
})
instr = m.input("instr", fields=INSTR)               # 传 spec 则 width 可省略

op   = instr["opcode"]                 # 下标形式
rd   = instr.rd                        # 属性形式
pair = instr["opcode", "rd"]           # 多字段拼接读

cnt = domain.signal(name="cnt", width=32, fields={"opcode": (31, 26), "rd": (25, 21)})
cnt <<= cnt + 1                        # 算术 / <<= 委托给底层信号
nxt = cnt.update(rd=new_rd)            # 读-改-写，结果仍是绑定信号
m.output("rd", wire_of(cnt.rd))        # wire_of / m.output 自动解包
```

`BitfieldSpec.bind(sig)`（即 `fields=` 背后的机制）返回 **`BitfieldSignal`**：它把算术、比较、`<<=`、位切片全部委托给底层信号（drop-in 替换），只额外增加字段访问。`x.raw` 取回底层信号，`x.spec` 取回布局。

**访问语义速查**

| 写法 | 含义 |
|------|------|
| `x["opcode"]` / `x.opcode` | 字段读（字符串下标 / 属性） |
| `x["opcode", "rd"]` | 多字段拼接读（MSB-first，展开为 `cat`） |
| `x[3]` / `x[0:8]` | 原始位切片（整数 / `slice` 下标，Python 半开区间） |
| `x.update(rd=v)` | 读-改-写，返回新的绑定信号 |
| `x.raw` / `x.spec` | 底层信号 / 布局 |

规则：`BitfieldView` **只读**（写用 `spec.update()`）；单次 `update` 不得写重叠字段（歧义，报错）；字段值可为 `int`（带值域检查）/ `Wire` / CAS，宽度不符报错；字段名与包装成员冲突（如 `update`/`raw`/`width`/`spec`）时用字符串下标访问。

> **纯前端语法糖**：字段读展开为 `pyc.extract`，`update` 展开为 `pyc.concat`，与手写 slice/cat 的 MLIR **字节级等价**。不改方言语义。

### 3.9 位掩码模式匹配（`matches` / `in_`）

把指令编码表写成与架构手册一字对照的形式，展开为 `(sig & mask) == value`。

```python
hit  = opcode.matches("1xx0")               # ≡ (opcode & 0b1001) == 0b1000
hit2 = opcode.matches("1(0)x0")             # 括号内位也忽略，与 "1xx0" 等价
sel  = opcode.in_(["000x", "0010", "11xx"]) # 任一命中（or 归约）
miss = opcode.not_in_(["1xx0", "0011"])     # ASL IN !{...}

is_add = instr["opcode"].matches("100000")  # 与 §3.8 位域视图配合
```

**模式语法**（MSB-first）：

| 字符 | 含义 |
|------|------|
| `0` / `1` | 关心位（必须相等） |
| `x` / `X` / `-` | 忽略位 |
| `(...)` | 括号内每一位都是忽略位（无论写 0 还是 1），对应 ASL `'1(0)x0'` |
| 空格 / `_` | 分隔符，仅为可读性，不占位宽 |

模式位数必须**等于信号宽度**，否则展开期报错；`in_` / `not_in_` 接受变长参数或单个可迭代对象，空集合报错。结果一律是 i1。

### 3.10 类型安全枚举（`PycEnum`）

枚举是**命名类型**，不与裸整数互转——防「拼错成员 / 宽度不够 / 跨枚举比较」。

```python
from pycircuit import PycEnum, auto, enumeration

# 写法① ASL 式一行声明（只列名字）
SRType = enumeration("SRType", "LSL LSR ASR ROR")     # 也可传列表或多个参数

# 写法② class 形式（需要显式编码 / 文档字符串时用）
class SRType(PycEnum):
    LSL = auto(); LSR = auto(); ASR = auto(); ROR = auto()

SRType.width        # 2 == ceil(log2(4))，由成员数自动推导
SRType.ASR.value    # 2（成员编码；基于 enum.Enum 而非 IntEnum，不隐式转 int）
```

声明即绑定，返回 **`EnumSignal`**：

```python
op  = m.input("op", enum=SRType)        # width 自动 = SRType.width
hit = op.is_(SRType.LSL)                # ≡ (raw == 0)，返回 i1
m.output("is_ror", op == SRType.ROR)    # == 是 is_ 的别名

st = domain.signal(name="state", enum=SRType)
st <<= SRType.LSR                       # 枚举寄存器加载成员编码

tagged = SRType.bind(existing_sig)      # 给已有信号贴枚举类型
c      = SRType.ROR.const(m)            # 定宽常量（domain 参数则返回 CAS）
raw    = op.raw                         # 取回底层信号（位级运算走 .raw）
```

**类型安全边界**（以下均在展开期 `TypeError`）：

```python
Color = enumeration("Color", "RED GREEN BLUE")   # 另一个枚举类型

op.is_(0)            # 裸 int → 报错（枚举不与整数互转）
op == 2              # 同上
op.is_(Color.RED)    # 跨枚举 → 报错
```

`EnumSignal` 刻意只暴露 `is_` / `is_not` / `==` / `!=`（仅接受同枚举成员）与 `.raw`；显式 `width=` 与 `E.width` 不一致时报错，且 `enum=` 不可与 `fields=` / `shape=` 混用。

### 3.11 受检转换、缩放切片与显式签名比较

**受检转换 `as_`**（对应 ASL `expression as ty`）——把「我确信它装得下 / 取值受限」的设计意图变成可仿真验证的契约。三类约束**互斥**，一次一种：

```python
# ① 取值集合（值直接作为位置参数，最省字）
y = x.as_(2)                     # 断言 x == 2，原样返回 x
y = x.as_(0, 5, 10)              # 断言 x ∈ {0,5,10}；也可 as_([0,5,10]) / as_(values=[...])
y = x.assert_in([0, 5, 10])      # 具名方法

# ② 值区间
y = x.as_(range=(2, 9))          # 断言 2 <= x <= 9（无符号），原样返回
y = x.assert_range(2, 9)         # 具名方法

# ③ 宽度收窄
y = x.as_(width=4)               # 断言高位为 0，再 trunc 到 i4
y = x.assert_fits(width=4)       # 等价别名
y = x.as_(width=4, msg="opcode 必须 4 位")
```

宽度收窄发射的 MLIR（i8 → i4）：

```mlir
%hi = pyc.extract %x {lsb = 4, msb = 7} : i8 -> i4
%ok = pyc.eq %hi, 0 -> i1
pyc.assert %ok {msg = "as_: value does not fit in 4 bits"}
%y  = pyc.trunc %x : i8 -> i4        // ← 综合路径只保留这一步
```

| 规则 | 说明 |
|------|------|
| 位置参数只归取值断言 | `x.as_(2)` 明确表示「断言 == 2」，不会被当成 2 位宽；`width=` / `range=` 必须用关键字 |
| 断言可丢弃 | `pyc.assert` 是仿真期副作用，综合时丢掉即退化为纯 `trunc`（零硬件代价契约） |
| 取值 / 区间断言不改数据路径 | 只挂契约，宽度与值原样返回 |
| 等宽 / 加宽 | `as_(width=x.width)` 是 no-op；加宽报错（改用 `zext` / `sext`） |
| 平凡边界省略 | `range=` 的边界若被宽度天然满足则不发对应比较，全覆盖时不发断言 |
| 有符号收窄 | 当前只断言无符号意义的「高位全零」，有符号收窄请显式配合 `as_signed()` |

与静默 `trunc()` 的区别：`trunc` 悄悄砍高位，`as_` 留下一条可仿真验证的契约。

**缩放切片 `lane`**（对应 ASL `x[idx *: len]`）——按元素粒度索引扁平总线：

```python
lane = bus[i, 8]                 # 下标糖（元组 (index, width)），最接近 ASL x[i *: 8]
lane = bus.lane(i, width=8)      # 等价函数式写法
                                 # 两者 ≡ bus[i*8 : (i+1)*8]；lane i 占 bit [i*8, i*8+7]
```

`i` / `width` 为展开期 Python `int`；`width <= 0`、`i < 0`、越界（`i*width+width > 信号宽度`）均在展开期报错。元组下标 `x[i, w]` 与位切片 `x[a:b]`、位域多字段读 `x["a","b"]` 无歧义（下标类型不同）。

> ASL 的 `*:` 在 Python 中是语法错误，故用元组下标编码。多 lane 同构数据的首选仍是 `Wire[Vector]`（§3.3）；`lane` 面向已被打平成 packed 总线的残余场景。

**显式签名比较**——`<` / `>` 的签名取决于操作数的 `signed` 标记，容易误判。签名不能从上下文一眼看出的比较，一律用显式方法：

| 无符号 | 有符号 |
|--------|--------|
| `ult` / `ugt` / `ule` / `uge` | `slt` / `sgt` / `sle` / `sge` |

`Wire` 与 CAS 都提供全套八个方法，结果为 i1，分别落到 `pyc.ult` / `pyc.slt`。

**编程约定速记**：**比较显式签名、截断显式宽度、常量集中分层**。前两条分别由本节的显式比较与 `as_(width=)` 落实；第三条指架构常量按意图分层（派生量 vs 参数量）集中一处定义。

---

## 4. Forward Signal 与寄存器推导

**核心思想：先声明后赋值，编译器根据读写周期差推导寄存器。**

```python
# 1. 声明（cycle 0）：Q 端立即可读
counter = domain.signal(width=8, reset_value=0, name="counter")

# 2. 组合逻辑（cycle 0）
count_next = mux(enable, counter + 1, counter)
m.output("count", wire_of(counter))

# 3. 推进周期
domain.next()                    # → cycle 1

# 4. 赋值：写周期 1 > 读周期 0 → 推导出一级反馈寄存器
counter <<= count_next
```

### 周期推导规则

| 读周期 | 写周期 | 推导结果 |
|--------|--------|----------|
| 0 | 1 | 一级反馈寄存器（DFF）——最常见 |
| 0 | 0 | 组合赋值（无寄存器） |
| 0 | N≥2 | N 级流水反馈（罕见，需明确意图） |
| N | < N | **编译错误**：不能向过去赋值 |

---

## 5. 全局函数

### cas()

```python
x = cas(domain, m.input("x", width=8), cycle=0)
```

将裸 `Wire`（来自 `m.input()` / `m.const()`）包装为 CAS。是端口层与 CAS 类型系统之间的唯一桥梁。

### mux()

```python
result = mux(condition, true_value, false_value)
```

三个参数为 CAS（或 int 字面量），返回 CAS，自动周期对齐。也接受标量 / 向量 `Wire` 操作数（分别产生标量 / 逐 lane mux）。

### wire_of()

```python
m.output("result", wire_of(outs["result"]))
```

从 CAS / `ForwardSignal` / 裸 `Wire` / `Reg` 中提取裸 `Wire`。**仅**允许出现在 `m.output()` 调用中。

### submodule_input()

双模输入解析：

```python
pc = submodule_input(inputs, "pc", m, domain, prefix="fe", width=32)
```

| `inputs` 状态 | 行为 |
|---------------|------|
| `None`（独立模式） | 创建 `m.input(f"{prefix}_{key}", width=W)` 并 `cas()` 包装 |
| dict 含 `key` | 直接返回 `inputs[key]`（父模块 CAS，cycle 保留） |
| dict 不含 `key` | 回退创建端口（通常是 key 拼写错误的征兆——务必传全） |

参数：`io, key, m, domain, *, prefix, width, cycle=0`。

---

## 6. 周期管理与自动周期平衡

### next() / prev()

```python
domain.next()   # 逻辑周期 +1，标记时序分界点
domain.prev()   # 回退，用于补充同周期逻辑
```

### push() / pop() / call()

`domain.call()` 自动包裹 push/pop，保证**周期隔离**：

```python
print(domain.cycle_index)          # 0
domain.next()                      # → 1
child_out = domain.call(child_fn, inputs={...})   # 子函数内部任意 next()
print(domain.cycle_index)          # 1 ← 恢复，不受子函数影响
```

隔离由 push/pop 栈 + `try/finally` 保证（子函数抛异常也会恢复）。**注意**：隔离的是父函数的周期计数器，**不改变**返回信号自身携带的 cycle 值（cycle provenance）。

### 自动周期平衡

组合不同周期的信号时：**输出周期 = max(输入周期)**，较早的信号自动插入 DFF 延迟链。

```python
# sig_a 在 cycle 0，sig_b 在 cycle 2
result = sig_a + sig_b
# → result 在 cycle 2；sig_a 自动延迟 2 拍
```

生成的 MLIR：

```mlir
%a_d1 = pyc.reg %clk, %rst, %en, %a,    %init : i8
%a_d2 = pyc.reg %clk, %rst, %en, %a_d1, %init : i8
%r    = pyc.add %a_d2, %b : i8
```

---

## 7. 模块签名与层次化组合

### 7.1 标准模块签名

每个 V6 模块是一个普通 Python 函数：

```python
def my_module(
    m: CycleAwareCircuit,          # ① 共享电路对象
    domain: CycleAwareDomain,      # ② 共享时钟域
    *,
    inputs: dict | None = None,    # ③ None=独立模式；dict=被组合
    width: int = 64,               # ④ 配置参数（keyword-only）
    prefix: str = "mod",           # ⑤ 端口/寄存器名前缀
) -> dict:                          # ⑥ 输出信号字典（值为 CAS）
    ...

my_module.__pycircuit_name__ = "my_module"   # 注册 RTL 模块名
```

### 7.2 双模运行

| 模式 | `inputs` | 输入 | 输出 | 用途 |
|------|----------|------|------|------|
| 独立 | `None` | `m.input()` 创建端口 | `m.output()` 发射端口 | 单元测试 / 独立综合 |
| 组合 | `{...}` | 从父模块 dict 读取 CAS | 仅返回 dict | 集成到父模块 |

### 7.3 子模块调用六步法

1. **声明自身输入**：`_in = submodule_input; a = _in(inputs, "a", m, domain, prefix=prefix, width=W)`
2. **构造子模块 inputs dict**：key 必须与子模块 `submodule_input()` 的 key **完全一致**，值必须是 CAS
3. **调用**：`child_out = domain.call(child_fn, inputs={...}, **config, prefix=f"{prefix}_ch")`
4. **读取输出**：`child_out["result"]`——CAS，cycle 保留子模块内部赋值时的值
5. **级联**：前一个子模块的输出直接作为下一个子模块的输入
6. **顶层收集**：`if inputs is None: m.output(f"{prefix}_{k}", wire_of(v))`

### 7.4 命名约定

前缀层次级联，避免冲突：

| 元素 | 模式 | 示例 |
|------|------|------|
| 输入 / 输出端口 | `{prefix}_{name}` | `fe_bpu_pc` |
| 状态寄存器 | `{prefix}_{name}` | `fe_fetch_pc` |
| 子模块前缀 | `{parent_prefix}_{child}` | `soc_cpu_fe` |

### 7.5 常见错误

| 错误 | 后果 | 纠正 |
|------|------|------|
| dict 值传 `Wire` 而非 CAS | 丢失 cycle 信息 / 类型错误 | 先 `cas()` 包装 |
| 输出 dict 存 `wire_of(x)` | 父模块无法周期对齐 | 存 CAS 本体 |
| key 拼写不匹配 | 静默创建多余端口 | key 与子模块 `_in` 完全一致 |
| 子模块共用 prefix | 端口 / 寄存器名冲突 | 每个 call 独立 prefix |
| 独立模式忘记 `m.output()` | 逻辑被 DCE 删光 | `if inputs is None:` 分支发射输出 |

### 7.6 层次化 MLIR 发射

```python
# 扁平（默认）：单一 func.func
circ = compile_cycle_aware(top, eager=True, name="top")

# 层次化：每个 domain.call() 边界保留为独立模块
circ = compile_cycle_aware(top, eager=True, name="top", hierarchical=True)
```

层次化模式下每个子模块编译为独立 `func.func`，父模块发射 `pyc.instance` 引用；输出的 MLIR 为多模块 `Design`（`module attributes {pyc.top = @top}`）。子模块内部的 `domain.call()` 递归处理。

---

## 8. 存储 / FIFO / CDC 原语

`Circuit`（`CycleAwareCircuit` 继承）提供以下原语，直接映射到 `pyc` 方言 op，并由后端提供匹配的 Verilog 原语模块与 C++ 模型：

| 前端方法 | MLIR op | 语义 |
|----------|---------|------|
| `m.fifo(...)` | `pyc.fifo` | 单时钟 ready/valid FIFO（attr `depth`） |
| `m.byte_mem(...)` | `pyc.byte_mem` | 异步读、同步写、字节使能存储 |
| `m.sync_mem(...)` | `pyc.sync_mem` | 同步 1R1W（读数据打拍） |
| `m.sync_mem_dp(...)` | `pyc.sync_mem_dp` | 同步 2R1W |
| `m.async_fifo(...)` | `pyc.async_fifo` | 双时钟异步 FIFO |
| `m.cdc_sync(...)` | `pyc.cdc_sync` | 多级同步器（attr `stages`，默认 2） |
| `m.rv_queue(...)` | 组合原语 | ready/valid 队列；`pop()` 返回 `Pop(valid, data, fire)` |

跨时钟域数据**必须**经 `cdc_sync` / `async_fifo`；`pyc-check-clock-domains` 强制检查。

---

## 9. 仿真与测试（Tb / CycleAwareTb / Sidecar）

### 9.1 Tb（周期编号模型）

```python
from pycircuit import Tb, testbench

@testbench
def tb(t: Tb) -> None:
    t.clock("clk")
    t.reset("rst", cycles_asserted=2, cycles_deasserted=1)
    t.timeout(64)
    t.drive("enable", 1, at=1)
    t.expect("count", 1, at=2, phase="post")
    t.finish(at=10)
```

- `phase="pre"`：时钟沿计算后、提交前观测（TICK-OBS）
- `phase="post"`（默认）：提交后观测（XFER-OBS）
- 其余 API：`print` / `print_every` / `sva_assert` / `random`

### 9.2 CycleAwareTb（隐式周期推进）

将 `at=cycle` 替换为与设计对称的 `tb.next()`：

```python
from pycircuit import CycleAwareTb, Tb, testbench

@testbench
def tb(t: Tb) -> None:
    tb = CycleAwareTb(t)
    tb.clock("clk")
    tb.reset("rst", cycles_asserted=2, cycles_deasserted=1)
    tb.timeout(64)

    tb.drive("enable", 0)
    tb.expect("count", 0)

    tb.next()                  # → cycle 1
    tb.drive("enable", 1)
    tb.expect("count", 0)      # 时钟沿后才更新

    tb.next()                  # → cycle 2
    tb.expect("count", 1)
    tb.finish()
```

| 方法 | 说明 |
|------|------|
| `CycleAwareTb(t)` | 包装 `Tb` |
| `next()` / `cycle` | 推进 / 读当前周期 |
| `drive(port, value)` | 当前周期驱动 |
| `expect(port, value, *, phase="post", msg=None)` | 当前周期检查 |
| `finish(*, at=None)` / `print(...)` / `timeout(n)` | 结束 / 打印 / 超时 |
| `clock` / `reset` / `sva_assert` / `random` | 透传 `Tb` |

### 9.3 测试调度：inline vs sidecar

测试事件（drive/expect）有两种下发方式（`pycircuit build --tb-schedule-mode {inline,sidecar}`）：

| 模式 | 机制 | 适用 |
|------|------|------|
| `inline`（默认） | 事件编入生成的 C++ 源码 | 短测试 |
| `sidecar` | 事件序列化为二进制 **SIDECAR 容器**，运行时由稳定 runner 加载 | 长测试（避免 C++ 编译膨胀；改激励不必重编） |

Sidecar 容器（魔数 `SIDECAR\n`）含五个 section：`string_table`、`port_table`、`event_table`、`frame_table`、`pattern_table`。周期性反压（如 ready 口按固定模式起伏）自动压缩为 `periodic_drive` pattern。

检查工具：

```bash
pycircuit sidecar inspect out/tb.sidecar        # 打印容器内容
pycircuit sidecar verify  out/tb.sidecar        # 校验结构
```

---

## 10. 编译入口

### 10.1 compile_cycle_aware()（V6 主路径）

```python
def compile_cycle_aware(
    fn,
    *,
    name: str | None = None,       # 模块名
    domain_name: str = "clk",      # 时钟域名
    eager: bool = False,           # True=直接执行 fn；False=JIT 追踪
    hierarchical: bool = False,    # True=保留 domain.call() 边界（需 eager=True）
    **jit_params,                  # 转发给 fn 的配置参数
)
```

```python
circ = compile_cycle_aware(my_module, name="my_module", eager=True, width=16)
mlir_text = circ.emit_mlir()
```

- `eager=True`：直接执行 Python 函数体，即时构图。**推荐路径**。支持任意 Python 控制流（作为元编程展开）。
- `eager=False`（JIT）：AST 解析 fn，不执行；支持把 Python `if`（i1 条件）编译为 `scf.if` → mux，`for`（静态可迭代）展开。有原型级限制。
- `hierarchical=True`：见 §7.6。

### 10.2 @module JIT 路径（V4 风格，仍受支持）

```python
from pycircuit import module, compile, Circuit

@module
def build(m: Circuit): ...

design = compile(build)
```

`@module` 边界产生 `pyc.instance`；`@function` 内联；`@const` 为编译期纯元编程（禁发 IR）。`pycircuit emit` / `pycircuit build` CLI 走此入口。

### 10.3 CLI

```bash
# 生成 MLIR
pycircuit emit design.py -o design.pyc [--param k=v ...]

# 一键构建 + 仿真（多 .pyc → pycc → CMake/Verilator）
pycircuit build tb_design.py --out-dir build/ \
    [--target {cpp,verilator,both}] [--jobs N] [--profile {dev,release}] \
    [--logic-depth N] [--tb-schedule-mode {inline,sidecar}] \
    [--run-verilator] [--param k=v ...]

# Sidecar 工具
pycircuit sidecar inspect FILE [--strict]
pycircuit sidecar verify FILE
```

---

## 11. MLIR 映射参考

前端构图产物为 `pyc` 方言 MLIR（`.pyc` 文件）。主要映射：

### 类型

| 前端 | MLIR |
|------|------|
| `Bits(W)` / 标量信号 width=W | `iW` |
| `Vector(N, Bits(W))`（shape=[N]，width=W） | `vector<NxiW>` |
| `Clock` / `Reset` | `!pyc.clock` / `!pyc.reset` |

### 运算（标量与逐元素向量共用）

| 前端 | MLIR op |
|------|---------|
| `+ - *` | `pyc.add / sub / mul` |
| `// %`（按符号） | `pyc.udiv / urem / sdiv / srem` |
| `& \| ^ ~` | `pyc.and / or / xor / not` |
| `mux / select` | `pyc.mux` |
| `== < （按符号）` | `pyc.eq / ult / slt` |
| `trunc / zext / sext` | `pyc.trunc / zext / sext` |
| `[lo:hi]`（标量）、`lane(i, width=)`、`x[i, w]` | `pyc.extract`（attrs `lsb`、可选 `msb`） |
| `<< >>`（常量位移） | `pyc.shli / lshri / ashri` |
| `<< >>`（动态位移） | `pyc.shl / lshr / ashr` |
| `cat(...)` | `pyc.concat`（MSB-first） |
| `.named()` | `pyc.alias`（attr `pyc.name`） |

### 状态与结构

| 前端 | MLIR op |
|------|---------|
| `domain.signal()` + `<<=` / 自动平衡 DFF | `pyc.reg %clk, %rst, %en, %next, %init` |
| 可赋值 wire | `pyc.wire` + `pyc.assign` |
| `domain.call()`（层次化） | `pyc.instance`（attr `callee`） |
| FIFO / mem / CDC | `pyc.fifo / byte_mem / sync_mem / sync_mem_dp / async_fifo / cdc_sync` |
| 复位有效视图 | `pyc.reset_active` |
| 断言 | `pyc.assert` |

### 向量

| 前端 | MLIR op |
|------|---------|
| `v[i]` | `pyc.v_get` |
| `m.vec(...)` | `pyc.v_create` |
| 标量广播 | `pyc.v_broadcast` |
| `broadcast(dim=)` | `pyc.v_broadcast_dim` |
| `reduce_or / reduce_and / reduce_sum` | `pyc.v_or_reduce / v_and_reduce / v_add_reduce`（attrs `dim`, `mode`） |

### 类型级语法糖（§3.8–§3.11）

这些特性**不引入新 op**，全部在展开期降解到既有 op——与手写等价代码的 MLIR 字节级一致：

| 前端 | 降解为 |
|------|--------|
| `spec.view(x)["fld"]` / `x.fld` / `x[i, w]` / `lane()` | `pyc.extract`（attrs `lsb`、可选 `msb`） |
| `spec.update(x, fld=v)` / `x["a","b"]` | `pyc.concat`（MSB-first）+ `pyc.extract` |
| `x.matches(p)` | `pyc.and` + `pyc.eq`（`(x & mask) == value`） |
| `x.in_(...)` / `x.not_in_(...)` | 各 `matches` 的 `pyc.or` 归约（`not_in_` 再取 `pyc.not`） |
| `E.MEMBER.const(m)` / `x.is_(E.M)` | `pyc.constant` + `pyc.eq` |
| `x.as_(width=w)` | `pyc.extract` + `pyc.eq` + `pyc.assert` + `pyc.trunc` |
| `x.as_(range=)` / `x.as_(values)` | `pyc.ult` / `pyc.eq` 组合 + `pyc.assert`（数据路径不变） |
| `sgt / sle / sge` | `pyc.slt`（+ `pyc.not`） |

`pyc.extract` 的 `msb` 是**可选**属性；存在时 verifier 强制 `msb == lsb + result_width - 1`，仅为阅读 MLIR 时一眼看出提取范围，旧 IR（无 `msb`）仍合法。`pyc.assert` 在 Verilog 侧包在 `` `ifndef SYNTHESIS`` 中，综合可丢弃。

测试台不产生硬件 op：`@testbench` 程序序列化为模块属性 `pyc.tb.payload`（inline 模式）或外置 sidecar 容器。

---

## 12. Tier 分层标注（3D 堆叠扩展，Proposed）

> **状态:Proposed**(尚未实现;完整提案与实现草图见 `docs/rfcs/tier_annotation.md`)。本节先行纳入规范,冻结语法形态与语义边界。

面向 3D 堆叠(细粒度逻辑折叠:设计折叠到 2–3 层垂直堆叠的裸片)的源码级层指派。信号携带第二种元数据 **`.tier`**(裸片层号),与 `.cycle` 并列。

**术语纪律:分层维度一律用 tier(裸片层),不用 layer**(后者指金属布线层)。

### 12.1 与 `.cycle` 的语义对照(理解本扩展的钥匙)

| | `.cycle` | `.tier` |
|---|---|---|
| 语义地位 | **行为语义**,切错位置行为就变 | **物理提示**,不产生/不修改任何硬件 |
| 传播规则 | max 规则 + 自动周期平衡(插 DFF) | 继承规则,零电路效应 |
| 验证影响 | 必须过功能等价验证 | 逻辑恒等,功能验证无感 |

两套传播机制共存于同一次 elaboration,互不干扰。

### 12.2 语法

```python
a  = cas(domain, m.input("a", width=8), tier=0)          # 定义处显式
pc = domain.signal(width=64, name="pc", tier=0)           # 前向声明处显式
s1 = a + 1                                                # 隐式:继承输入的 tier
s2 = jump_tier(s1 * 3, to=1)                              # 强制跳层(声明一个键合点)
hot = cas(domain, m.input("b", width=8), tier=1,
          tier_lock=True)                                 # 锁定:下游 EDA 不得改写
tmp = cas(domain, m.input("c", width=8))                  # 未指派:tier=None,EDA 全权
outs = domain.call(alu, inputs={...}, tier=1)             # 模块级缺省 tier
```

### 12.3 语义规则

1. **传播:** 运算结果的 tier 由输入继承(全部/多数同层继承之;混层取主导方向),强度记为推断;
2. **反馈信号:** `domain.signal()` 的 tier 在声明处确定,`<<=` 赋值不改变它;跨层反馈须用 `jump_tier` 显式表达;
3. **自动周期平衡插入的对齐 DFF 继承驱动信号的 tier**(平衡不引入额外跨层);
4. **`jump_tier(expr, to=k)`:** 返回新 CAS,`.tier == k`、`.cycle` 不变、零硬件效应;`to` 须为编译期常量;
5. **强度三态与 EDA 契约:** free(未指派,分割器全权)/ hint(显式与推断,可改写但必须输出结构化 diff)/ locked(必须服从,不可满足报错)。分割器结果写入以稳定 ID 为键的 sidecar tier 表,不回写源码。

### 12.4 IR 与发射

- MLIR 可选属性:`pyc.tier`(int)、`pyc.tier_strength`(`"hint"|"locked"`)、模块级 `pyc.tier_default`;无标注即无属性,**完全向后兼容**;
- Verilog 三条冗余通道:`(* pyc_tier = 1 *)` 属性、层次化命名编码、sidecar tier 指派表(主通道);三者不一致构成流程告警;
- C++ 仿真后端忽略 tier(功能无关),可选地在 DFX 元数据中携带以便按层聚合统计。

---

## 13. 附录：API 参考表

### CycleAwareCircuit

| 方法 | 说明 |
|------|------|
| `CycleAwareCircuit(name)` | 创建顶层电路 |
| `create_domain(name, ...)` | 时钟域 |
| `input(name, *, width, signed=False, shape=None, fields=None, enum=None)` | 输入端口（`shape=None` → `Wire[Bits]`；`shape=[N]` → `Wire[Vector[...]]`；`fields=` → `BitfieldSignal`；`enum=` → `EnumSignal`） |
| `output(name, value)` | 输出端口 |
| `const(value, *, width)` | 常量 |
| `vec(*elems)` / `cat(parts)` | 容器构造 |
| `fifo / byte_mem / sync_mem / sync_mem_dp / async_fifo / cdc_sync / rv_queue` | 原语 |
| `emit_mlir()` | 导出 MLIR |

### CycleAwareDomain

| 方法 | 说明 |
|------|------|
| `signal(*, width, reset_value=0, name="", shape=None, fields=None, enum=None)` | 前向声明寄存器（唯一方式）；`fields=` → `BitfieldSignal`，`enum=` → `EnumSignal` |
| `cycle(sig, ...)` | 单级 DFF |
| `next()` / `prev()` | 推进 / 回退周期 |
| `push()` / `pop()` | 周期栈 |
| `call(fn, *, inputs=None, **kwargs)` | 子模块调用（自动隔离） |
| `delay_to(w, *, from_cycle, to_cycle, width)` | 显式打拍 |
| `create_signal / create_const / create_reset` | 端口 / 常量 / 复位 |
| `cycle_index` | 当前逻辑周期 |

### 全局函数

| 函数 | 说明 |
|------|------|
| `cas(domain, wire, cycle=N)` | Wire → CAS |
| `mux(cond, t, f)` | 多路选择（自动对齐） |
| `submodule_input(io, key, m, domain, *, prefix, width, cycle=0)` | 双模输入 |
| `wire_of(sig)` | 提取 Wire（仅 `m.output()`） |
| `cat / zext / sext / trunc` | 位操作辅助 |

### ForwardSignal

| 接口 | 说明 |
|------|------|
| `sig <<= expr` | 无条件赋值 |
| `sig.assign(expr, when=cond)` | 条件赋值（使能） |
| 其余读侧接口 | 与 CAS 相同 |

### Wire[Vector]（向量专属方法）

| 接口 | 说明 |
|------|------|
| 逐 lane 运算符 | `+ - * & \| ^ ~ == != < >` 等；标量自动广播 |
| `reduce_or / reduce_and (*, dim=None, mode="chain")` | 位归约；`dim=None` 全维成标量，`dim=int` 降一维 |
| `reduce_sum(*, dim=None, mode="chain")` | 求和归约；**保持叶宽，溢出回绕**（需加宽先用 `zext`/`sext`） |
| `broadcast(*, size, dim)` | 维度广播 |
| `priority_mux(sels, vals, *, mode="chain", default=None)` | lane 选择（`sels` 为 i1 选择器向量，最小索引优先；`default=None` 时回退到 `vals` 末元素）；模块级函数或 CAS 实例方法 |
| `m.cat(*lanes)` / `[i]` | 拼成 packed 位向量 / 取 lane |

### 类型级语法糖（`Wire` 与 CAS 通用）

| 接口 | 说明 |
|------|------|
| `BitfieldSpec(width=, fields={name: (msb, lsb)})` | 位域布局；`(msb, lsb)` 闭区间，允许字段重叠 |
| `spec.view(x)` / `spec(x)` | 只读视图；`f["fld"]` / `f.fld` / `f["a","b"]` |
| `spec.update(x, fld=v)` | 读-改-写，返回同类型信号 |
| `spec.bind(x)` / `m.input(fields=)` / `domain.signal(fields=)` | 声明即绑定 → `BitfieldSignal`（drop-in + 字段访问 + `.update()` + `.raw` / `.spec`） |
| `spec.field_slices()` / `spec.field_width(n)` | 布局查询（名 → `(lsb, width)` / 字段宽度） |
| `x.matches(p)` / `x.in_(ps)` / `x.not_in_(ps)` | 位掩码匹配 / 任一命中 / 均不命中（i1） |
| `enumeration("E", "A B C")` / `class E(PycEnum)` | 类型安全枚举（0-based 编码，`E.width` 自动） |
| `m.input(enum=E)` / `domain.signal(enum=E)` / `E.bind(x)` | 声明即绑定 → `EnumSignal` |
| `x.is_(E.M)` / `x.is_not(E.M)` / `==` / `!=` | 成员比较（i1）；裸 int 与跨枚举报错 |
| `E.M.const(m/domain)` / `x.raw` | 定宽常量 / 取回底层信号 |
| `x.as_(v, …)` / `x.assert_in([...])` | 断言取值集合，原样返回 |
| `x.as_(range=(lo,hi))` / `x.assert_range(lo,hi)` | 断言无符号区间，原样返回 |
| `x.as_(width=w)` / `x.assert_fits(width=w)` | 受检收窄：断言高位为零 + `trunc(w)` |
| `x.lane(i, *, width)` / `x[i, w]` | 缩放切片（ASL `x[i *: w]`） |
| `ult / ugt / ule / uge`、`slt / sgt / sle / sge` | 显式无符号 / 有符号比较（全套） |

### compile_cycle_aware

| 参数 | 说明 |
|------|------|
| `fn` | `def fn(m, domain, *, inputs=None, ...) -> dict` |
| `name` / `domain_name` | 模块名 / 时钟域名 |
| `eager` | `True` 直接执行（推荐） |
| `hierarchical` | 保留 `domain.call()` 边界（需 eager） |
| `**jit_params` | 转发给 `fn` |

### CycleAwareTb

| 方法 | 说明 |
|------|------|
| `CycleAwareTb(t)` / `next()` / `cycle` | 包装 / 推进 / 读周期 |
| `drive(port, value)` / `expect(port, value, *, phase, msg)` | 激励 / 检查 |
| `clock / reset / timeout / finish / print / sva_assert / random` | 配置与控制 |

---

**Copyright © 2024-2026 Liao Heng / PyCircuit Contributors. All rights reserved.**
