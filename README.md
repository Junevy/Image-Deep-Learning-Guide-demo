# Python 学习项目

这是一个按学习路线整理的 Python 实践项目，内容从 Python 基础语法逐步扩展到文件处理、标准库、虚拟环境、Pillow、NumPy 科学计算、OpenCV 图像处理以及 PyTorch 深度学习实战。

## 目录概览

| 目录 | 内容 |
| --- | --- |
| [`01_Primary`](01_Primary) | Python 基础语法、函数、高阶函数、装饰器、生成器、面向对象、枚举、异常和单元测试 |
| [`02_io`](02_io) | 字符串 IO、字节 IO、序列化与反序列化（Marshal、JSON） |
| [`03_inner_module`](03_inner_module) | `datetime`、`namedtuple`、`Counter`、迭代器等 Python 标准库常用模块 |
| [`04_venv`](04_venv) | Python 虚拟环境相关实验 |
| [`05_pillow`](05_pillow) | Pillow 基础图像处理练习 |
| [`06_numpy`](06_numpy) | NumPy 数组基础、索引、变形、拼接、广播、矩阵运算、随机数和综合练习 |
| [`07_opencv`](07_opencv) | OpenCV 图像读取、显示、保存、像素与 ROI、通道处理、几何变换、滤波形态学、模板匹配与特征拼接 |
| [`08_pytorch`](08_pytorch) | PyTorch 张量基础、Dataset/DataLoader、CNN 网络搭建、TensorBoard 可视化、模型保存加载、ONNX 导出与分类训练流程 |

## 推荐学习顺序

```text
Python 基础 (01_Primary)
    ↓
文件与标准库 (02_io / 03_inner_module)
    ↓
虚拟环境与图像基础 (04_venv / 05_pillow)
    ↓
NumPy 科学计算 (06_numpy)
    ↓
OpenCV 图像处理 (07_opencv)
    ↓
PyTorch 深度学习 (08_pytorch)
```

建议先在 `01_Primary` 巩固语法，再阅读 `02_io` 与 `03_inner_module`；完成 `06_numpy` 的数组运算后，进入 `07_opencv` 进行计算机视觉图像操作，最后通过 `08_pytorch` 迈向深度学习建模与训练。

---

## 环境与运行

### 1. 虚拟环境配置

针对不同阶段的依赖，建议配置或选用对应的虚拟环境：

- **通用 Python / NumPy**：项目根目录 `.venv` 或各子目录环境，Python 3.14
- **OpenCV 练习**：`07_opencv/.venv`，内核选择 **OpenCV (Python 3.14.7)**
- **PyTorch 深度学习**：推荐使用 Conda 独立的 PyTorch 环境（如 `venv_pytorch`），需要安装 `torch`、`torchvision`、`tensorboard` 等。

### 2. 运行方式

在 VS Code / PyCharm 中打开项目根目录：
- 直接运行 `.py` 脚本；
- 运行 `.ipynb` Notebook 时，右上角切换为对应模块匹配的 Python 内核；
- 启动 Jupyter Notebook：
  ```powershell
  .venv\Scripts\jupyter.exe notebook
  ```

---

## 模块详细介绍

### OpenCV 模块 ([`07_opencv`](07_opencv))

- 基础入门位于 [`07_opencv/01_start`](07_opencv/01_start)，包含 02–16 号独立 Notebook，拆分讲解读写、像素、ROI、通道、缩放旋转、阈值与滤波等；
- 进阶练习位于 [`07_opencv/02_excrcise`](07_opencv/02_excrcise)：
  - `01_laplacian.py`：Laplacian 方差评估图像清晰度；
  - `02_template_match.ipynb`：模板匹配与多角度旋转模板；
  - `03_match_test.py`：带掩膜（Mask）的旋转匹配；
  - `04_img_joint.py`：SIFT 特征提取与图像拼接；
  - `05_rgb_to_gray.ipynb`：彩色图像转灰度图原理与实现对比；
- 统一输入图片存放于 [`07_opencv/img`](07_opencv/img)；
- 完整知识点总结请参阅 [`07_opencv/summary.md`](07_opencv/summary.md) 与 [`07_opencv/01_start/README.md`](07_opencv/01_start/README.md)。

### PyTorch 深度学习模块 ([`08_pytorch`](08_pytorch))

本模块系统梳理了从张量基础到神经网络搭建、训练、调优与工程化导出的全流程：

1. **基础语法与张量 ([`00_start_base_grammar`](08_pytorch/00_start_base_grammar))**
   - `01_start.py`：PyTorch 安装与版本/设备检验；
   - `02_tensor.ipynb`：Tensor 创建、属性、切片、数学运算以及与 NumPy 数组的无缝互转。
2. **核心组件构建 ([`01_base`](08_pytorch/01_base))**
   - `01_base_function.ipynb`：常用内置张量函数；
   - `02_datasets.ipynb` / `03_define_dataset.py`：官方数据集加载与自定义 Dataset 类的编写规范；
   - `04_tensorboard.py`：利用 TensorBoard 记录标量（Loss/Acc）与可视化模型图；
   - `05_transform.ipynb`：`torchvision.transforms` 图像变换、归一化与数据增强；
   - `06_dataloader.ipynb`：`DataLoader` 批量批次切分、洗牌与数据迭代；
   - `07_conv_demo.ipynb` / `08_define_CNN.py`：卷积操作原理与简单 CNN 网络构建；
   - `09_max_pooling.py` / `10_non-linear_active.py` / `12_linear.ipynb`：池化层、激活函数（ReLU/Sigmoid）与全连接线性层；
   - `11_cnn_trainng.ipynb`：端到端完整的卷积神经网络训练实战。
3. **实战应用与导出 ([`02_attached`](08_pytorch/02_attached))**
   - `01_datasets.ipynb` & `02_classification_train.ipynb`：工业缺陷分类任务实战，涵盖数据集处理、模型构建、验证评估；
   - `03_export_model.ipynb`：将训练好的 PyTorch 深度学习模型转换为 ONNX 格式以便部署推理；
   - `04_small_cnn.ipynb`：轻量化卷积网络探索。
4. **模型管理与操作 ([`03_model_operations`](08_pytorch/03_model_operations))**
   - `01_load_vgg_16.py`：经典预训练模型（如 VGG16）的下载、加载与分类头修改迁移学习；
   - `02_save.py` / `03_load.py`：模型完整序列化保存与基于 `state_dict` 字典参数保存的两种推荐方式与加载恢复。
5. **规范训练流程 ([`04_train_flow`](08_pytorch/04_train_flow))**
   - `model.py`：解耦的网络架构定义（如 `TenClassNet`）；
   - `01_base_flow.py`：工程化标准训练模板（设备自动分配、训练步迭代、反向传播优化、验证集评估）。

---

## 数据集与大文件管理规范

为了保证 Git 仓库的轻量和秒级 Push 体验，并遵守 GitHub 单文件不超过 100 MB 的规范：

- **数据集**：所有本地数据集目录（如 `data/`、`datasets/`、`08_pytorch/data/`）均已配置在 [`.gitignore`](.gitignore) 中。代码中使用官方数据集（如 MNIST）时，脚本已开启 `download=True`，首次运行会自动拉取至本地；
- **模型权重与导出**：大模型权重（`*.pth`、`*.pt`）与导出文件（`*.onnx`、`*.onnx.data`）不进入版本管理；
- **训练日志**：TensorBoard 生成的日志文件（`*tfevents*`）以及 `runs/`、`steel_summary/` 等目录均已被忽略。

---

## 隐私与提交规范

仓库不应提交本机绝对路径、密钥、令牌、个人数据或生成文件。示例代码使用项目相对路径；运行产物和常见凭据文件已加入 [`.gitignore`](.gitignore)。如果敏感内容已经推送到 GitHub，请先立即撤销或轮换凭据，再按下方的历史清理流程处理提交记录。

### 已推送历史的处理顺序

1. 如果内容是密码、Token 或 API Key，先在对应服务中撤销并轮换；仅删除 Git 文件不能让已泄露凭据失效。
2. 在仓库的独立镜像克隆中安装 `git-filter-repo`，使用 `--path` 删除文件，或使用 `--replace-text` 替换历史文本：

   ```powershell
   git clone --mirror https://github.com/OWNER/REPOSITORY.git
   cd REPOSITORY.git
   py -m pip install git-filter-repo
   git filter-repo --sensitive-data-removal --replace-text C:/path/to/replacements.txt
   git push --force --mirror origin
   ```

3. 强制推送会改变提交 SHA，其他克隆必须重新克隆或按 `git-filter-repo` 的清理步骤处理，不能直接 `pull` 后再推送。
4. GitHub 的 Push/Audit 事件记录本身不会因为重写分支而消失；如果是凭据等敏感数据，重写后还要向 [GitHub Support](https://support.github.com/) 申请清理缓存、旧提交引用和受影响的 Pull Request。

历史重写属于破坏性操作，执行前应确认仓库所有者、需要保留的分支/标签和要替换的具体文本。
