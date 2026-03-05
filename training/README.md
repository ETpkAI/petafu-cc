# 🐾 宠物阿福 — LoRA 微调指南

## 快速开始

### 1. 准备训练数据

```bash
cd /Volumes/kuozhan/petafu/training
python prepare_data.py
```

这会从 `knowledge_base/` 的兽医学文献中提取 QA 对，输出到 `dataset/train.jsonl`。

### 2. LoRA 微调

**方式 A: Google Colab（推荐，免费GPU）**

在 Colab Notebook 中运行：

```python
!pip install unsloth
!git clone https://github.com/ETpkAI/PetAfu.git
%cd PetAfu/training
!python prepare_data.py
!python train_lora.py
```

**方式 B: 本地 GPU**

```bash
pip install unsloth
python train_lora.py
```

> 需要 NVIDIA GPU，至少 8GB 显存（0.5B 模型）

### 3. 导出 GGUF

```bash
python export_gguf.py
```

输出：`output/petafu-vet-0.8b-q4km.gguf`（约 400MB）

### 4. 本地测试

```bash
# 安装 Ollama
brew install ollama

# 导入自定义模型
cat > Modelfile << 'EOF'
FROM ./output/petafu-vet-0.8b-q4km.gguf
SYSTEM "你是宠物健康AI助手'宠物阿福'。基于兽医学文献提供参考性建议。禁止确诊性语言。"
PARAMETER temperature 0.3
PARAMETER num_ctx 1024
EOF

ollama create petafu-vet -f Modelfile
ollama run petafu-vet "我的猫最近老是打喷嚏，怎么回事？"
```

## 切换模型大小

```bash
# 0.8B（默认，适合旧手机）
BASE_MODEL=Qwen/Qwen3.5-0.5B-Instruct python train_lora.py

# 2B（性能好的手机）
BASE_MODEL=Qwen/Qwen3.5-1.5B-Instruct python train_lora.py
```

## 文件说明

| 文件                  | 说明                                  |
| --------------------- | ------------------------------------- |
| `prepare_data.py`     | 从知识库提取 QA 训练数据              |
| `train_lora.py`       | LoRA 微调脚本（使用 unsloth）         |
| `export_gguf.py`      | 导出为 GGUF 格式                      |
| `dataset/train.jsonl` | 训练数据（自动生成）                  |
| `output/`             | 训练输出（LoRA 权重、合并模型、GGUF） |
