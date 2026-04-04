# 实验报告：基于 torchvision 的预训练 CNN 图像分类

## 一、实验目的

1. 使用 PyTorch 与 `torchvision.models` 加载在 ImageNet 上预训练的典型图像分类网络，并可通过 `print(model)` 查看实现细节。
2. 读入本地示例图像，在与预训练一致的预处理下进行前向推理（本报告 **不展开逐张图像的预测结果**，侧重结构与模型对比）。

## 二、实验环境

- **硬件**：实验运行时可选用 CUDA（若可用）或 CPU。
- **软件**：Python 3，PyTorch 2.7，torchvision 0.22（具体版本以本机 `torch.__version__` / `torchvision.__version__` 为准）。
- **代码位置**：`kaggle_code/cnn_fine_pred.py`（当前脚本主要实例化 AlexNet、VGG16、GoogLeNet、ResNet18；**NiN** 在经典文献与课程实现中常见，torchvision 标准模型列表中一般不直接提供 ImageNet 版 NiN，下文仍按原理论述以便与 AlexNet/VGG 等对照）。

## 三、原理简述

1. **预训练权重**：AlexNet、VGG16、GoogLeNet、ResNet18 等可使用 torchvision 提供的 ImageNet-1K 预训练权重枚举；首次运行会从官方地址下载权重并缓存。
2. **输入与归一化**：与 ImageNet 迁移学习常用设定一致，采用 `Resize(256)` → `CenterCrop(224)` → `ToTensor()` → 按通道 `Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])`。
3. **推理**：`model.eval()`，`torch.no_grad()` 前向得到 1000 维 logits；若需可读类名，可由权重元数据中的 `categories` 与 `argmax` / TOP-K 对应。

## 四、网络结构与多模型对比（合并说明）

本部分将「各网络在代码中打印出的模块级结构」与「AlexNet、VGG、NiN、GoogLeNet、ResNet 之间的设计差异」合并说明：先在表 1 作总览，再对各网络分述，最后归纳对比维度。

**表 1  五类经典 CNN 结构特点总览**

| 模型 | 核心卷积形态 | 深度/宽度特点 | 分类头（末端） | 代表性设计动机 |
|------|--------------|----------------|----------------|----------------|
| AlexNet | 大卷积核（如 11×11）与多层卷积 + 池化交替 | 相对浅、通道设计粗犷 | 多层大型全连接（如 4096）+ Dropout | 证明深度 CNN + ReLU + Dropout 在大规模数据上可训练并领先传统方法 |
| VGG | 几乎全部 **3×3** 卷积重复堆叠 | 深而规整，参数量大 | 展平后大型全连接（如 4096×2）+ Dropout | 用小卷积加深网络以增大有效感受野，结构简单、易堆叠 |
| NiN | 常规卷积 + **两层 1×1 卷积** 组成的「MLP 卷积」块 | 块内通道混合、跨像素共享少 | **全局平均池化** + 可选 1×1 映射类别，**避免传统全连接** | 用 1×1 增强逐位置非线性；GAP 减轻过拟合、压缩参数量 |
| GoogLeNet | **Inception 模块**（多分支 1×1 / 3×3 / 5×5 + 池化再拼接） | 深、但在模块内用 1×1 降维控制计算 | 全局平均池化 + 线性分类（训练期可有 **辅助分类头**） | 多尺度并行提取特征，在准确率和算力之间折中 |
| ResNet | 堆叠 **残差块**（恒等映射 + 卷积分支） | 可训练极深网络（如 18/34/50/101/152） | 全局平均池化 + 单层线性（1000 类） | 残差学习缓解极深网络优化困难与表示退化 |

---

### 4.1 AlexNet

- **整体形态**：前半为卷积 + 池化构成的特征提取器（`features`），末尾用自适应平均池化把特征图固定到较小空间尺寸（如 6×6），再送入多层全连接分类器（`classifier`）。
- **直观差异**：相对 VGG，AlexNet 更「宽核、大步幅」起步（如第一层 11×11、stride 4），通道数阶梯上升（64→192→384→256→256），符合早期 GPU 显存与并行设计习惯。
- **与 NiN / GoogLeNet / ResNet 对比**：AlexNet **依赖厚重全连接层** 聚合全局信息；NiN 与 GoogLeNet 倾向用 **GAP 或结构化模块** 弱化全连接；ResNet 则普遍 **单层线性** 接在 GAP 之后，参数集中在卷积残差路径上。

---

### 4.2 VGG（以 VGG-16 为代表）

- **整体形态**：`features` 中为重复的 **3×3 卷积 + ReLU**，每隔若干层用 **2×2 MaxPool** 减半空间分辨率；通道数按阶段递增（64、128、256、512、512）。末端 `AdaptiveAvgPool2d((7,7))` 或与训练时一致的展平维度，再接 **两个 4096** 维全连接与 **1000** 维输出。
- **与 AlexNet 差异**：VGG **放弃了大卷积核**，用多个 3×3 串联达到与大核相当的有效感受野，**结构高度同质**，便于加深与复现；代价是 **参数量和计算量显著高于** 同代许多网络（尤其全连接部分）。
- **与 NiN 差异**：VGG 仍采用 **大矩阵全连接** 作主要分类器；NiN 用 **1×1 卷积 + GAP** 替代或减少全连接，参数量与过拟合风险通常更可控。
- **与 GoogLeNet / ResNet 差异**：VGG 是「一条主干的顺序卷积栈」；GoogLeNet 在每个 Inception 内 **多分支并行**；ResNet 在每个块内 **残差相加**，使梯度更易贯穿极深层。

---

### 4.3 NiN（Network in Network）

- **核心思想**：在普通卷积层之后叠加 **两层 1×1 卷积**（与 ReLU），形成对每个空间位置的「小 MLP」，从而在**不改变感受野**的前提下增强 **逐点非线性** 与 **跨通道混合**。
- **全局平均池化（GAP）**：在最后将通道数为类别数（或经 1×1 映射到类别数）的特征图做 **整张图的空间平均**，每个通道对应一类，再经 `Flatten` 得到 logits。**传统大型全连接层可被去掉或极大弱化**，这是与 AlexNet、VGG 在 **结构哲学上最明显的不同**。
- **与 GoogLeNet 的隐约联系**：GoogLeNet 广泛使用的 **1×1 卷积瓶颈** 与 NiN 类似，都强调 **用 1×1 做通道维度的非线性变换与降维**；但 GoogLeNet 的主干是 Inception 多尺度并联，NiN 主干仍是规则卷积块 + 池化的堆叠。
- **与 ResNet 差异**：NiN 解决的是「分类头参数过多」问题；ResNet 主要解决 **极深网络的优化与表示退化**，两者可独立叠加思想（深层网络中仍可大量使用 1×1 与 GAP）。

---

### 4.4 GoogLeNet（Inception v1）

- **Inception 模块**：在同一层上 **并行** 多个不同卷积与池化分支（典型地经 1×1 降维后再做 3×3、5×5），再在通道维 **拼接**，使网络 **单层即可看到多尺度特征**，而不必把所有层都加深。
- **1×1 卷积的作用**：除 NiN 式的非线性外，这里更突出 **降低通道数、减少计算量**，使「宽而深」的网络在算力上可行。
- **辅助分类器**：训练时在中继层引出 **额外的分类支路**（加权重叠加到总损失），缓解梯度在极深结构中的传到问题；推理阶段通常 **可忽略** 这些头。AlexNet / VGG / 标准 ResNet **无** 这一设计。
- **末端**：通常 **全局平均池化** 后接 **线性层** 到 1000 类，与 VGG 的「双 4096」相比 **更轻量**。

---

### 4.5 ResNet（以 ResNet-18 为浅层代表）

- **残差块**：输出为 **F(x)+x**（或投影后的 x）：网络显式学习残差映射 F，若最优映射接近恒等，则 F→0 即可，**降低极深网络的学习难度**。
- **与 AlexNet / VGG**：后两者是「plain」堆叠，层数过深时易出现优化问题；ResNet 用 **跳连** 使梯度有捷径可回传，因此可出现 **18、34、50、101、152** 等实用深度，而不仅限于十来层。
- **分类头**：标准实现中在 `avgpool` 后对特征向量做 **单层 Linear(512→1000)**（具体维度随变体而变，如 ResNet-50 为 2048 维），**没有像 VGG 那样两层 4096**；参数主要留在卷积骨干。
- **与 GoogLeNet**：两者都能训练较深网络；GoogLeNet 靠 **Inception 模块的宽度与多尺度**，ResNet 靠 **残差恒等路径的深度**；现代网络常 **融合** 二者思想（如 Inception-ResNet 等，本报告不展开）。

---

### 4.6 横向归纳：阅读 `print(model)` 时可对照的维度

1. **卷积核形态**：AlexNet 有大核起步；VGG 统一小核深栈；NiN 强调 1×1；GoogLeNet 在模块内多尺度；ResNet 在块内通常是 3×3（及 1×1 瓶颈块）。
2. **分类器**：AlexNet / VGG 以 **大型全连接** 为标志；NiN / GoogLeNet / ResNet 普遍走向 **GAP + 少量线性** 或 **无传统 FC**。
3. **深度与训练技巧**：GoogLeNet 用 **辅助损失**；ResNet 用 **残差与 BatchNorm**（及后续变体中的各类归一化）；VGG/AlexNet 时代更依赖 Dropout 与数据增强等经典手段。
4. **工程与迁移学习**：五类网络均可作为 **特征提取骨干**；实际选用时常在 **精度、显存、延迟、微调难易** 之间权衡，而非单看 TOP-1。

**与实验流程的对应**：在 `cnn_fine_pred.py` 中执行 `print(alexnet)`、`print(vgg16)` 等，可将本节文字与终端中的 `Sequential`、`Conv2d`、`Linear`、`AdaptiveAvgPool2d`、`BasicBlock`/`Bottleneck`（视版本而定）等模块 **逐项对照**，加深对「同质深栈 / 多分支 / 残差 / 1×1+GAP」等差异的直观认识。

## 五、实验步骤

1. 安装/确认已安装 `torch` 与 `torchvision`。
2. 将待测图像放入 `data/images/`（路径与脚本中列表一致即可）。
3. 在项目根目录执行：

   ```bash
   python3 kaggle_code/cnn_fine_pred.py
   ```

4. 终端查看：各模型的 **模块级结构**（`print` 输出）；若需推理，可在代码中保留 TOP-K 逻辑，本报告不记录逐图预测数值。

## 六、分析与讨论（与结构相关）

1. **从 AlexNet/VGG 到 GAP + 浅分类头**：全连接层参数量随特征图展平维度爆炸式增长；NiN 与后续 GoogLeNet、ResNet 把「空间聚合」交给 **全局池化**，把「类别判别」交给 **少量线性参数**，有利于 **正则化** 与 **部署体积**。
2. **从单路径到多分支与残差**：GoogLeNet 的 Inception 与 ResNet 的 shortcut 均改善 **梯度流与特征表达**，但切入点不同：**并行多尺度** vs **恒等捷径稳定训练**。
3. **前处理一致性**：无论何种骨干，使用 ImageNet 预训练权重时都应保持 **一致的输入尺寸与归一化**，否则卷积统计特性与后续线性层不匹配，会扭曲输出分布。

## 七、结论

本实验/报告侧重视觉分类领域五条经典脉络：**AlexNet** 开启深度卷积实用化，**VGG** 用小核深栈展示「简单结构的可扩展性」，**NiN** 用 1×1 与全局平均池化 **弱化全连接**，**GoogLeNet** 用 Inception **在单层内融合多尺度**，**ResNet** 用残差 **把网络推向极深仍可训练**。在 `cnn_fine_pred.py` 中打印的结构可与上述差异一一对照，形成从「厚重全连接时代」到「GAP + 残差/Inception 时代」的完整认识；具体测试图像上的分类数值本报告不作展开。

## 八、参考文献与资料

- PyTorch 官方文档：<https://pytorch.org/docs/stable/index.html>
- torchvision.models：<https://pytorch.org/vision/stable/models.html>
- Krizhevsky et al., ImageNet Classification with Deep Convolutional Neural Networks (AlexNet).
- Simonyan & Zisserman, Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG).
- Lin, Chen, Yan, Network in Network (NiN).
- Szegedy et al., Going Deeper with Convolutions (GoogLeNet / Inception).
- He et al., Deep Residual Learning for Image Recognition (ResNet).

---

*报告对应脚本：`kaggle_code/cnn_fine_pred.py`*
