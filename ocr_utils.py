from statistics import median


def order_ocr_results(ocr_results_unordered, panels):
    try:
        boxes = ocr_results_unordered["boxes"]
        texts = ocr_results_unordered["texts"]
        img_path = ocr_results_unordered.get("img_path", "")
    except Exception:
        raise ValueError("ocr_results_unordered 结构错误")

    if len(boxes) != len(texts):
        raise ValueError("boxes 与 texts 数量不一致")

    if not panels:
        return ocr_results_unordered

    n = len(boxes)

    # ---------- box centers & widths ----------
    box_centers = []
    box_widths = []

    for b in boxes:
        x1, y1, x2, y2 = b
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        box_centers.append((cx, cy))
        box_widths.append(abs(x2 - x1))

    # ---------- threshold ----------
    if len(box_widths) == 0:
        threshold = 0
    else:
        threshold = median(box_widths) * 0.2

    # ---------- panel centers ----------
    panel_centers = []
    for p in panels:
        px1, py1, px2, py2 = p
        pcx = (px1 + px2) / 2
        pcy = (py1 + py2) / 2
        panel_centers.append((pcx, pcy))

    # ---------- intersection ----------
    def intersection_area(box, panel):
        x1, y1, x2, y2 = box
        px1, py1, px2, py2 = panel

        ix1 = max(x1, px1)
        iy1 = max(y1, py1)
        ix2 = min(x2, px2)
        iy2 = min(y2, py2)

        if ix2 <= ix1 or iy2 <= iy1:
            return 0
        return (ix2 - ix1) * (iy2 - iy1)

    # ---------- assign boxes to panels ----------
    panel_to_boxes = {i: [] for i in range(len(panels))}

    for i in range(n):
        cx, cy = box_centers[i]

        candidate_panels = []

        for pid, p in enumerate(panels):
            px1, py1, px2, py2 = p
            if px1 <= cx <= px2 and py1 <= cy <= py2:
                candidate_panels.append(pid)

        if len(candidate_panels) == 1:
            panel_to_boxes[candidate_panels[0]].append(i)
            continue

        if len(candidate_panels) > 1:
            best_panel = None
            best_area = -1

            for pid in candidate_panels:
                area = intersection_area(boxes[i], panels[pid])
                if area > best_area:
                    best_area = area
                    best_panel = pid

            panel_to_boxes[best_panel].append(i)
            continue

        # no panel contains center → nearest panel center
        best_panel = None
        best_dist = float("inf")

        for pid, (pcx, pcy) in enumerate(panel_centers):
            dist = (cx - pcx) ** 2 + (cy - pcy) ** 2
            if dist < best_dist:
                best_dist = dist
                best_panel = pid

        panel_to_boxes[best_panel].append(i)

    # ---------- sort inside each panel ----------
    ordered_indices = []

    for pid in range(len(panels)):

        idxs = panel_to_boxes[pid]

        if not idxs:
            continue

        items = []

        for idx in idxs:
            cx, cy = box_centers[idx]
            items.append((idx, cx, cy))

        # sort by cx desc
        items.sort(key=lambda x: -x[1])

        columns = []

        for item in items:
            idx, cx, cy = item

            placed = False

            for col in columns:
                if abs(cx - col["mean_cx"]) <= threshold:
                    col["items"].append(item)

                    col["mean_cx"] = sum(i[1] for i in col["items"]) / len(col["items"])
                    placed = True
                    break

            if not placed:
                columns.append({"mean_cx": cx, "items": [item]})

        # sort columns right -> left
        columns.sort(key=lambda c: -c["mean_cx"])

        for col in columns:
            col["items"].sort(key=lambda x: x[2])
            for item in col["items"]:
                ordered_indices.append(item[0])

    # ---------- build result ----------
    ordered_boxes = [boxes[i] for i in ordered_indices]
    ordered_texts = [texts[i] for i in ordered_indices]

    return {"img_path": img_path, "boxes": ordered_boxes, "texts": ordered_texts}


# 绘制

import os
import cv2


def draw_boxes_and_print_texts(
    json,
    output_dir,
    box_color=(0, 255, 0),  # 绿色
    text_color=(0, 0, 255),  # 红色
    box_thickness=1,
    font_scale=0.6,
    font_thickness=2,
):
    """
    使用 boxes 绘制矩形框，并打印序号-文本
    """
    img_path = json["img_path"]
    boxes = json["boxes"]
    texts = json["texts"]

    assert len(boxes) == len(texts), "boxes 与 texts 数量不一致"

    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"无法读取图片: {img_path}")

    h, w = image.shape[:2]

    # === 终端打印 ===
    print(f"\nImage: {img_path}")
    for idx, text in enumerate(texts, start=1):
        print(f"[{idx}] {text}")

    # === 图像绘制 ===
    for idx, box in enumerate(boxes, start=1):
        x1, y1, x2, y2 = box

        # 防越界
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        # 画矩形框
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            box_color,
            box_thickness,
        )

        # 序号位置：左上角略上
        label = str(idx)
        (tw, th), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            font_thickness,
        )

        tx = x1
        ty = y1 - 4
        if ty - th < 0:
            ty = y1 + th + 4

        cv2.putText(
            image,
            label,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA,
        )

    # === 保存 ===
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, os.path.basename(img_path))
    cv2.imwrite(out_path, image)

    print(f"\nSaved to: {out_path}")
