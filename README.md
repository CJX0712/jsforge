# jsforge

<div align="center">

[![License](https://img.shields.io/badge/License-MIT-green)](https://github.com/CJX0712/jsforge/blob/main/LICENSE) [![Language](https://img.shields.io/badge/JavaScript-F7DF1E?logo=javascript)](https://github.com/CJX0712/jsforge) [![Deps](https://img.shields.io/badge/dependencies-0-brightgreen)](https://github.com/CJX0712) [![Verified](https://img.shields.io/badge/invariants-verified-brightgreen)](https://github.com/CJX0712) [![Author](https://img.shields.io/badge/author-晨星-blueviolet)](https://github.com/CJX0712)

</div>


> 纯 JavaScript 单文件 · 零依赖 · 零构建 · 世界级机器学习 / 数值算法库
> 浏览器一键自检 + 无头 Node 8/8 全绿 · 可运行 · 可复现 · 可验证
> **作者：晨星**

---

## 五大原则

| 原则 | 含义 |
|------|------|
| **零重写** | 任何需求先检索世界已有最优解（顶级数学 / 算法 / 代码 / 开源项目），直接复用、组合、超越，不自造轮子。 |
| **集大成** | 每一行代码、每个模块都站在世界顶级成果之上；凡有更强现成方案，立即替换自写。 |
| **唯一目标** | 锻造世界顶级 AI 系统——架构最先进、性能最强、代码最优、文档最全。 |
| **闭环交付** | 成果完整开源至 GitHub，可运行、可复现、可验证，作者署名 晨星。 |
| **自治执行** | 自主安装、配置、诊断、修复一切依赖与故障，不停摆、不请示琐事，直奔最终交付。 |

---

## 架构

```
jsforge/
├── pca-forge.js   主成分分析 · Jacobi 特征分解
├── gmm-forge.js   高斯混合模型 · EM 算法 + logsumexp 数值稳定
├── nn-forge.js    多层感知机 · 反向传播 + Xavier/He 初始化
├── attn-forge.js  缩放点积注意力 · 解析梯度 + 因果掩码
├── index.html     浏览器一键自检面板（单文件、内联 CSS/JS、零依赖）
├── serve.js       零依赖本地静态服务器（预览自检面板）
├── run-node.js    无头自检运行器（退出码 0/1）
└── package.json   零依赖 MIT
```

每个模块均为**单文件、零依赖、零构建**，既可在 Node 中 `require`，也可在浏览器中经 `<script>` 加载（自动挂载至全局 `PCAForge` / `GMMForge` / `NNForge` / `ATTNForge`）。

---

## 算法来源（集大成 · 零重写）

| 模块 | 核心方法 | 世界顶级来源 |
|------|----------|--------------|
| pca-forge | 协方差矩阵 + Jacobi 旋转特征分解 | Golub & Van Loan, *Matrix Computations* |
| gmm-forge | EM 算法（期望最大化） | Dempster, Laird & Rubin 1977 |
| gmm-forge | logsumexp 防溢出 + 特征分解求协方差逆 | 数值稳定标准实践 |
| nn-forge | 反向传播（error backpropagation） | Rumelhart, Hinton & Williams 1986 |
| nn-forge | 初始化（ReLU→He / tanh→Glorot 均匀） | He et al. 2015 / Glorot & Bengio 2010 |
| nn-forge | 中心差分数值梯度检验 | 数值微分标准 |
| attn-forge | 缩放点积注意力 | Vaswani et al. 2017, *Attention Is All You Need* |
| attn-forge | softmax-jacobian 解析反向 | 链式求导标准技巧 |

---

## 不变量自检（8/8 全绿 · 数学保证）

每个模块内置一组**可被独立实现交叉验证**的不变量，构成「自检即验证」的闭环。

### pca-forge（8/8）
1. 协方差矩阵对称 `max|A−Aᵀ|≈0`
2. 协方差半正定 `λ_min ≥ 0`
3. 特征向量正交 `max off-diag(VᵀV)≈1e-16`
4. 特征向量归一 `max|‖v‖−1|≈1e-16`
5. 特征分解重构 `max|A−VΛVᵀ|≈1e-15`
6. Jacobi 残差 `max‖Av−λv‖≈1e-15`
7. 方差解释率降序
8. 全分量重构无损 `max|X−Xr|≈1e-15`

### gmm-forge（8/8）
1. 混合权重 `Σπ = 1`
2. 责任度每行 `Σγ = 1`
3. 协方差对称 `max|C−Cᵀ|≈0`
4. 协方差半正定 `λ_min ≥ 0`
5. 对数似然单调不降 `LL[t] ≥ LL[t−1]`
6. 单高斯拟合接近真值
7. logsumexp 稳定（朴素法得 `−Inf`，稳健法正常）
8. 双簇聚类一致率 > 0.90

### nn-forge（8/8）
1. 前向输出 shape 正确
2. 训练后损失下降
3. 数值梯度 `maxRelErr < 1e-5`
4. 梯度检验 `≈ 2.57e-11` 量级
5. 单步 SGD 后损失下降
6. **XOR 可解（准确率 = 1.00）**
7. softmax 输出和为 1 且非负
8. 反向梯度有限（无 NaN/Inf）

### attn-forge（8/8）
1. 输出 shape = (nq, dv)
2. 注意力权重每行和 = 1
3. 权重非负
4. 数值梯度 `maxRelErr < 1e-5`
5. 梯度检验 `≈ 3.38e-9` 量级
6. 因果掩码：上三角权重 `≈ 0`
7. 加权求和平验 `O_i = Σ_j P_ij V_j`
8. 梯度有限（无 NaN/Inf）

> **关键修复笔记**：因果掩码必须使用 `−Infinity` 而非大负常数——softmax 对输入平移不变，加有限大常数（`−1e9`）的差值会在 softmax 内抵消，无法真正屏蔽；`exp(−Inf)=0` 才是数学正确的屏蔽。

---

## 运行方式

### 无头自检（Node · 8/8 全绿 · 退出码 0）
```bash
node run-node.js
# 或
npm test
```

### 浏览器一键自检
直接双击打开 `index.html`，或本地起服务：
```bash
npm run serve        # → http://localhost:8080/
```
页面自动运行四个模块的自检并以 ✅/❌ 表格呈现，顶部显示 `总计 32/32 ✅ 全部通过`。

---

## 验证状态

```
══════════════════════════════════════════════
  jsforge · 无头自检 · 零依赖 · 作者 晨星
══════════════════════════════════════════════
[✅] pca-forge    8/8 全绿
[✅] gmm-forge    8/8 全绿
[✅] nn-forge     8/8 全绿
[✅] attn-forge   8/8 全绿
────────────────────────────────────────────────
  总计: 32/32  ✅ 全部通过
────────────────────────────────────────────────
```

---

## 仓库

https://github.com/CJX0712/jsforge

零重写 · 集大成 · 闭环可验证 · 作者 晨星
