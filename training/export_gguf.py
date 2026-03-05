"""
将合并后的模型导出为 GGUF 格式（llama.cpp 可直接加载）

用法:
  python export_gguf.py

需要安装: pip install llama-cpp-python
或使用 llama.cpp 的 convert 脚本
"""
import os
import subprocess
from pathlib import Path

MERGED_DIR = Path(__file__).parent / "output" / "petafu-vet-merged"
GGUF_OUTPUT = Path(__file__).parent / "output" / "petafu-vet-0.8b-q4km.gguf"

# 量化类型（Q4_K_M 是速度和质量的最佳平衡）
QUANT_TYPE = os.environ.get("QUANT_TYPE", "q4_k_m")


def export_with_unsloth():
    """使用 unsloth 直接导出 GGUF（推荐）"""
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("❌ unsloth 未安装，尝试用 llama.cpp convert 导出")
        return False

    print(f"📦 使用 unsloth 导出 GGUF...")
    print(f"  源模型: {MERGED_DIR}")
    print(f"  量化类型: {QUANT_TYPE}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(MERGED_DIR),
        max_seq_length=1024,
        load_in_4bit=False,
    )

    output_dir = GGUF_OUTPUT.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained_gguf(
        str(output_dir / "petafu-vet"),
        tokenizer,
        quantization_method=QUANT_TYPE,
    )

    print(f"\n✅ GGUF 模型已导出: {output_dir}")
    return True


def export_with_llamacpp():
    """使用 llama.cpp 的 convert 脚本导出"""
    print("📦 使用 llama.cpp convert 导出 GGUF...")
    print("  请确保 llama.cpp 已克隆并编译")

    llama_cpp_dir = os.environ.get("LLAMA_CPP_DIR", "../../llama.cpp")

    if not os.path.exists(f"{llama_cpp_dir}/convert_hf_to_gguf.py"):
        print(f"❌ 找不到 llama.cpp，请设置 LLAMA_CPP_DIR 环境变量")
        print(f"   git clone https://github.com/ggerganov/llama.cpp")
        return False

    # Step 1: 转换为 FP16 GGUF
    fp16_path = GGUF_OUTPUT.parent / "petafu-vet-f16.gguf"
    subprocess.run([
        "python3", f"{llama_cpp_dir}/convert_hf_to_gguf.py",
        str(MERGED_DIR),
        "--outfile", str(fp16_path),
        "--outtype", "f16",
    ], check=True)

    # Step 2: 量化
    quantize_bin = f"{llama_cpp_dir}/build/bin/llama-quantize"
    if not os.path.exists(quantize_bin):
        quantize_bin = f"{llama_cpp_dir}/llama-quantize"

    subprocess.run([
        quantize_bin,
        str(fp16_path),
        str(GGUF_OUTPUT),
        QUANT_TYPE.upper(),
    ], check=True)

    # 清理 FP16 中间文件
    fp16_path.unlink(missing_ok=True)

    print(f"\n✅ GGUF 模型已导出: {GGUF_OUTPUT}")
    return True


def main():
    print(f"🐾 宠物阿福模型导出 → GGUF")
    print(f"  量化类型: {QUANT_TYPE}")
    print()

    if not MERGED_DIR.exists():
        print(f"❌ 合并模型不存在: {MERGED_DIR}")
        print("   请先运行 train_lora.py")
        return

    if not export_with_unsloth():
        export_with_llamacpp()

    # 显示最终文件大小
    for gguf in GGUF_OUTPUT.parent.glob("*.gguf"):
        size_mb = gguf.stat().st_size / 1024 / 1024
        print(f"\n📊 {gguf.name}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
