# -*- coding: utf-8 -*-
"""
将 LabelMe 标注的 JSON 转换为 YOLO 检测格式的 txt 标签。

背景：
    LabelMe 保存的是绝对像素坐标 (x1, y1, x2, y2)，
    YOLO 检测要求每个目标一行：<class_id> <cx> <cy> <w> <h>，且数值归一化到 0~1。

规则：
    1. 只处理 rectangle / polygon 两类标注，polygon 取外接矩形（本数据集全是 rectangle）。
    2. 坐标按 imageWidth / imageHeight 归一化，并裁剪到 [0, 1]，避免越界框导致训练异常。
    3. 宽或高接近 0 的退化框直接跳过并告警。
    4. 同名 txt 默认不覆盖（--overwrite 可强制覆盖），原始 json 始终保持不动。

用法：
    python convert_labelme_to_yolo.py --root bvn --classes junevy
    python convert_labelme_to_yolo.py --root bvn --classes junevy --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# 参与转换的数据划分
DEFAULT_SPLITS = ("train", "val")

# 退化框阈值：归一化后的宽或高小于该值视为无效框
MIN_BOX_SIZE = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LabelMe JSON -> YOLO txt 标签转换")
    parser.add_argument("--root", default="bvn", help="数据集根目录，需包含 images/<split> 与 labels/<split>")
    parser.add_argument("--splits", default=",".join(DEFAULT_SPLITS), help="数据划分，逗号分隔，默认 train,val")
    parser.add_argument("--classes", default="junevy", help="类别名列表，逗号分隔，顺序即 class_id（从 0 开始）")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的同名 txt")
    parser.add_argument("--dry-run", action="store_true", help="只解析统计，不写文件")
    return parser.parse_args()


def shape_to_bbox(shape: dict) -> tuple[float, float, float, float] | None:
    """把单个 shape 转成 (xmin, ymin, xmax, ymax)；不支持的标注类型返回 None。"""
    shape_type = shape.get("shape_type", "rectangle")
    points = shape.get("points") or []

    if shape_type not in ("rectangle", "polygon") or not points:
        return None

    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def convert_one(json_path: Path, class_map: dict[str, int], image_size: tuple[int, int] | None) -> tuple[list[str], list[str]]:
    """转换单个 json，返回 (txt 行列表, 告警信息列表)。"""
    lines: list[str] = []
    warnings: list[str] = []

    with open(json_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)

    img_w = float(data.get("imageWidth") or 0)
    img_h = float(data.get("imageHeight") or 0)

    # json 内记录的尺寸与磁盘图片实际尺寸不一致时，归一化结果会整体偏移，必须提示
    if image_size and img_w and img_h and image_size != (int(img_w), int(img_h)):
        warnings.append(f"{json_path.stem}: json 记录尺寸 {(int(img_w), int(img_h))} 与图片实际尺寸 {image_size} 不一致")

    if img_w <= 0 or img_h <= 0:
        warnings.append(f"{json_path.stem}: 缺少有效的 imageWidth / imageHeight，已跳过")
        return lines, warnings

    for index, shape in enumerate(data.get("shapes") or []):
        label = str(shape.get("label", "")).strip()
        if label not in class_map:
            warnings.append(f"{json_path.stem}: 第 {index} 个标注的类别 '{label}' 不在类别表中，已跳过")
            continue

        bbox = shape_to_bbox(shape)
        if bbox is None:
            warnings.append(f"{json_path.stem}: 第 {index} 个标注类型 '{shape.get('shape_type')}' 暂不支持，已跳过")
            continue

        xmin, ymin, xmax, ymax = bbox

        # 归一化 + 裁剪到图像范围内，防止人工标注越界
        xmin = min(max(xmin, 0.0), img_w)
        xmax = min(max(xmax, 0.0), img_w)
        ymin = min(max(ymin, 0.0), img_h)
        ymax = min(max(ymax, 0.0), img_h)

        box_w = (xmax - xmin) / img_w
        box_h = (ymax - ymin) / img_h
        if box_w < MIN_BOX_SIZE or box_h < MIN_BOX_SIZE:
            warnings.append(f"{json_path.stem}: 第 {index} 个标注为退化框（w={box_w:.6f}, h={box_h:.6f}），已跳过")
            continue

        center_x = (xmin + xmax) / 2.0 / img_w
        center_y = (ymin + ymax) / 2.0 / img_h
        lines.append(f"{class_map[label]} {center_x:.6f} {center_y:.6f} {box_w:.6f} {box_h:.6f}")

    return lines, warnings


def main() -> int:
    args = parse_args()

    root = Path(args.root).resolve()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    class_map = {name.strip(): idx for idx, name in enumerate(args.classes.split(",")) if name.strip()}

    if not root.is_dir():
        raise SystemExit(f"数据集根目录不存在：{root}")
    if not class_map:
        raise SystemExit("类别表为空，请检查 --classes 参数")

    print(f"数据集根目录：{root}")
    print(f"类别映射：{class_map}")
    print(f"写入模式：{'dry-run（不写文件）' if args.dry_run else ('覆盖' if args.overwrite else '跳过已存在')}")

    total_json = total_written = total_skipped = total_boxes = 0
    all_warnings: list[str] = []

    for split in splits:
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            print(f"[{split}] 目录缺失，跳过：{image_dir} / {label_dir}")
            continue

        json_files = sorted(label_dir.glob("*.json"))
        written = skipped = boxes = 0

        for json_path in json_files:
            txt_path = json_path.with_suffix(".txt")
            total_json += 1

            if txt_path.exists() and not args.overwrite:
                skipped += 1
                continue

            # 用同名图片的真实尺寸做交叉校验
            image_size = None
            for suffix in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                image_path = image_dir / f"{json_path.stem}{suffix}"
                if image_path.exists():
                    try:
                        from PIL import Image

                        with Image.open(image_path) as img:
                            image_size = img.size
                    except Exception as exc:  # 图片损坏不影响标签转换，只记录
                        all_warnings.append(f"{json_path.stem}: 读取图片尺寸失败（{exc}）")
                    break

            lines, warnings = convert_one(json_path, class_map, image_size)
            all_warnings.extend(warnings)

            if not lines:
                # 没有目标时写空文件，YOLO 视其为纯背景图；不写则会被判定为“缺标签”
                all_warnings.append(f"{json_path.stem}: 无有效标注框，写入空 txt（按纯背景图处理）")

            if not args.dry_run:
                with open(txt_path, "w", encoding="utf-8", newline="\n") as fp:
                    fp.write("\n".join(lines) + ("\n" if lines else ""))

            written += 1
            boxes += len(lines)

        total_written += written
        total_skipped += skipped
        total_boxes += boxes
        print(f"[{split}] json {len(json_files)} 个，转换 {written} 个，跳过 {skipped} 个，生成框 {boxes} 个")

    print(f"合计：处理 json {total_json} 个，转换 {total_written} 个，跳过 {total_skipped} 个，生成框 {total_boxes} 个")

    if all_warnings:
        print(f"\n告警 {len(all_warnings)} 条：")
        for msg in all_warnings:
            print(f"  - {msg}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
