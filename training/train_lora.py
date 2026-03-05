"""
LoRA 微调 Qwen3.5 — 宠物阿福兽医助手

使用 unsloth 加速微调（如无 GPU 可在 Google Colab 上运行）
支持模型: Qwen/Qwen3.5-0.8B, Qwen/Qwen3.5-2B

用法:
  # 本地 GPU
  python train_lora.py

  # Google Colab (在 notebook 中)
  !pip install unsloth
  %run train_lora.py
"""
import os
import json
from pathlib import Path

# ── 配置 ──────────────────────────────────────────────────────────
# 模型选择：
#   - Qwen/Qwen2.5-0.5B-Instruct (推荐: 小巧、兼容性好)
#   - Qwen/Qwen3.5-0.8B (新架构，内存占用大)
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
DATASET_PATH = Path(__file__).parent / "dataset" / "train.jsonl"
OUTPUT_DIR = Path(__file__).parent / "output"
LORA_OUTPUT = OUTPUT_DIR / "petafu-vet-lora"
MERGED_OUTPUT = OUTPUT_DIR / "petafu-vet-merged"

# 训练超参数（针对 T4 15GB 优化）
EPOCHS = 3
BATCH_SIZE = 2          # 降低以节省内存
GRAD_ACCUM = 4          # 保持有效 batch size = 8
LEARNING_RATE = 2e-4
LORA_RANK = 16
LORA_ALPHA = 32
MAX_SEQ_LEN = 512       # 降低以节省内存


def load_dataset_from_jsonl(path: str):
    """加载 JSONL 格式的训练数据"""
    from datasets import Dataset

    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            # 将 messages 格式转为 text 格式
            messages = item["messages"]
            text = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    text += f"<|im_start|>system\n{content}<|im_end|>\n"
                elif role == "user":
                    text += f"<|im_start|>user\n{content}<|im_end|>\n"
                elif role == "assistant":
                    text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
            data.append({"text": text})

    return Dataset.from_list(data)


def main():
    print(f"🐾 宠物阿福 LoRA 微调")
    print(f"  基座模型: {BASE_MODEL}")
    print(f"  训练数据: {DATASET_PATH}")
    print(f"  LoRA Rank: {LORA_RANK}")
    print(f"  学习率: {LEARNING_RATE}")
    print(f"  Epochs: {EPOCHS}")
    print()

    # ── 1. 加载模型 ──
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("❌ 请先安装 unsloth: pip install unsloth")
        print("   或在 Google Colab 中运行: !pip install unsloth")
        return

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
    )

    # ── 2. 添加 LoRA 适配器 ──
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # ── 3. 加载数据 ──
    print(f"📂 加载训练数据...")
    dataset = load_dataset_from_jsonl(str(DATASET_PATH))
    print(f"  共 {len(dataset)} 条数据")

    # ── 4. 训练 ──
    from trl import SFTTrainer
    from transformers import TrainingArguments

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        args=TrainingArguments(
            output_dir=str(OUTPUT_DIR / "checkpoints"),
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LEARNING_RATE,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            weight_decay=0.01,
            fp16=True,
            logging_steps=10,
            save_strategy="epoch",
            seed=42,
            report_to="none",
        ),
    )

    print(f"\n🚀 开始训练...")
    trainer.train()

    # ── 5. 保存 LoRA 权重 ──
    LORA_OUTPUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(LORA_OUTPUT))
    tokenizer.save_pretrained(str(LORA_OUTPUT))
    print(f"\n✅ LoRA 权重已保存: {LORA_OUTPUT}")

    # ── 6. 合并权重 ──
    print(f"\n🔀 合并 LoRA 权重到基座模型...")
    MERGED_OUTPUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(
        str(MERGED_OUTPUT),
        tokenizer,
        save_method="merged_16bit",
    )
    print(f"✅ 合并模型已保存: {MERGED_OUTPUT}")


if __name__ == "__main__":
    main()
