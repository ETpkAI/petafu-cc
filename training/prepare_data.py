"""
从 knowledge_base 中的兽医学 Markdown 文件提取 QA 对
用于 LoRA 微调 Qwen3.5

提取策略：
1. 按标题切分为知识段落
2. 每个段落生成 instruction + input + output 格式的训练数据
3. 输出为 JSONL 格式
"""
import json
import re
import os
from pathlib import Path

KB_DIR = Path(__file__).parent.parent / "knowledge_base"
OUTPUT = Path(__file__).parent / "dataset" / "train.jsonl"

SYSTEM_MSG = (
    "你是宠物健康AI助手'宠物阿福'。基于兽医学文献为宠物主人提供参考性健康建议。"
    "禁止使用确诊性语言，用'疑似''可能'替代。回答结尾加免责声明。"
)


def extract_sections(md_text: str) -> list[dict]:
    """
    按 Markdown 标题 (##, ###) 切分为段落，
    每个段落生成一个 QA 对。
    """
    sections = []
    # 按 ## 或 ### 切分
    parts = re.split(r'\n(?=#{2,3}\s)', md_text)

    for part in parts:
        part = part.strip()
        if not part or len(part) < 50:  # 太短的段落跳过
            continue

        # 提取标题
        title_match = re.match(r'^#{2,3}\s+(.+)', part)
        if not title_match:
            continue

        title = title_match.group(1).strip()
        body = part[title_match.end():].strip()

        # 过滤掉纯图片引用的段落
        text_only = re.sub(r'!\[.*?\]\(.*?\)', '', body).strip()
        if len(text_only) < 30:
            continue

        # 截断过长内容（小模型上下文有限）
        if len(text_only) > 800:
            text_only = text_only[:800] + "..."

        sections.append({
            "title": title,
            "content": text_only,
        })

    return sections


def section_to_qa(section: dict) -> list[dict]:
    """将一个段落转为多种形式的 QA 对"""
    title = section["title"]
    content = section["content"]
    qa_pairs = []

    # 模式1: 直接提问标题
    qa_pairs.append({
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": f"请介绍一下{title}"},
            {"role": "assistant", "content": content + "\n\n⚠️ 本内容来源于兽医学术文献检索，仅供参考，不作为最终医疗判断依据。请前往正规宠物医院由执业兽医诊断。"},
        ]
    })

    # 模式2: 症状相关问题（如果标题包含症状/疾病关键词）
    disease_keywords = ["病", "症", "感染", "炎", "虫", "瘤", "损伤", "中毒", "缺乏"]
    if any(kw in title for kw in disease_keywords):
        qa_pairs.append({
            "messages": [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": f"我的宠物可能有{title}的症状，请帮我分析一下"},
                {"role": "assistant", "content": f"根据兽医学文献，关于{title}的相关信息如下：\n\n{content}\n\n⚠️ 以上内容仅供参考，建议尽快就医确认。"},
            ]
        })

    # 模式3: 治疗方案问题
    treatment_keywords = ["治疗", "用药", "处置", "手术", "预防"]
    if any(kw in title or kw in content[:100] for kw in treatment_keywords):
        qa_pairs.append({
            "messages": [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": f"{title}应该怎么治疗？"},
                {"role": "assistant", "content": f"关于{title}的处理建议（仅供参考）：\n\n{content}\n\n⚠️ 具体治疗方案请遵医嘱，本内容不作为医疗依据。"},
            ]
        })

    return qa_pairs


def main():
    all_qa = []
    md_files = list(KB_DIR.rglob("*.md"))

    print(f"📚 找到 {len(md_files)} 个知识库文件")

    for md_file in md_files:
        print(f"  处理: {md_file.name}")
        text = md_file.read_text(encoding="utf-8")
        sections = extract_sections(text)
        print(f"    提取到 {len(sections)} 个知识段落")

        for section in sections:
            qa_pairs = section_to_qa(section)
            all_qa.extend(qa_pairs)

    # 去重
    seen = set()
    unique_qa = []
    for qa in all_qa:
        key = qa["messages"][1]["content"]
        if key not in seen:
            seen.add(key)
            unique_qa.append(qa)

    # 保存
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for qa in unique_qa:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")

    print(f"\n✅ 共生成 {len(unique_qa)} 条训练数据")
    print(f"📁 保存到: {OUTPUT}")


if __name__ == "__main__":
    main()
