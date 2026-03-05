"""
LoRA 微调 Qwen2.5 — 宠物阿福兽医助手（纯 transformers 版本）

不依赖 unsloth，使用标准 transformers + peft
适用于 Colab T4 GPU

用法:
  pip install transformers peft datasets trl accelerate bitsandbytes
  python train_lora_simple.py
"""
import os
import json
import torch
from pathlib import Path

# ── 配置 ──────────────────────────────────────────────────────────
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
DATASET_PATH = Path(__file__).parent / "dataset" / "train.jsonl"
OUTPUT_DIR = Path(__file__).parent / "output"
LORA_OUTPUT = OUTPUT_DIR / "petafu-vet-lora"
MERGED_OUTPUT = OUTPUT_DIR / "petafu-vet-merged"

# 训练超参数（针对 T4 15GB 优化）
EPOCHS = 3
BATCH_SIZE = 1          # 降低以节省内存
GRAD_ACCUM = 8          # 保持有效 batch size = 8
LEARNING_RATE = 2e-4
LORA_RANK = 8           # 降低以节省内存
LORA_ALPHA = 16
MAX_SEQ_LEN = 512


def load_dataset_from_jsonl(path: str):
    """加载 JSONL 格式的训练数据"""
    from datasets import Dataset

    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
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
            # 截断过长的文本
            if len(text) > 1500:
                text = text[:1500]
            data.append({"text": text})

    return Dataset.from_list(data)


def main():
    print(f"🐾 宠物阿福 LoRA 微调 (纯 transformers)")
    print(f"  基座模型: {BASE_MODEL}")
    print(f"  训练数据: {DATASET_PATH}")
    print(f"  LoRA Rank: {LORA_RANK}")
    print(f"  学习率: {LEARNING_RATE}")
    print(f"  Epochs: {EPOCHS}")
    print()

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer

    # ── 1. 加载 tokenizer ──
    print("📥 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 2. 加载模型（4bit 量化）──
    print("📥 加载模型...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # ── 3. 配置 LoRA ──
    print("⚙️ 配置 LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── 4. 加载数据 ──
    print("📂 加载训练数据...")
    dataset = load_dataset_from_jsonl(str(DATASET_PATH))
    print(f"  共 {len(dataset)} 条数据")

    # ── 5. 训练 ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        weight_decay=0.01,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        seed=42,
        report_to="none",
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        args=training_args,
    )

    print("\n🚀 开始训练...")
    trainer.train()

    # ── 6. 保存 LoRA 权重 ──
    LORA_OUTPUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(LORA_OUTPUT))
    tokenizer.save_pretrained(str(LORA_OUTPUT))
    print(f"\n✅ LoRA 权重已保存: {LORA_OUTPUT}")

    print("\n🎉 训练完成！")


if __name__ == "__main__":
    main()
