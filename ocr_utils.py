from statistics import median


def get_ordered(unordered_ocr_result, panels):
    try:
        boxes = unordered_ocr_result["boxes"]
        texts = unordered_ocr_result["texts"]
        img_path = unordered_ocr_result.get("img_path", "")
    except Exception:
        raise ValueError("unordered_ocr_result 结构错误")

    if len(boxes) != len(texts):
        raise ValueError("boxes 与 texts 数量不一致")

    if not panels:
        return unordered_ocr_result

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


def get_ordered_list(unordered_ocr_results, results):
    if len(unordered_ocr_results) != len(results):
        raise ValueError("unordered_ocr_results 与 results 的（图片）数量不一致")

    return [
        get_ordered(unordered_ocr_result, result["panels"])
        for unordered_ocr_result, result in zip(unordered_ocr_results, results)
    ]


def print_texts(img_path, texts):
    """
    打印序号-文本
    """
    print(f"\nImage: {img_path}")
    for idx, text in enumerate(texts):
        print(f"[{idx}] {text}")


from torch.nn.utils.rnn import pad_sequence
import torch
from model.florence2.utils import UnionFind
from PIL import Image


@torch.no_grad()
def predict_with_injected_ocr(
    model,
    processor,
    img_paths,
    unordered_ocr_results,  # list of {"boxes": [[x1,y1,x2,y2],...], "texts": [...]}
    character_character_association_threshold=0.5,
    text_character_association_threshold=0.8,
    text_tail_association_threshold=0.8,
    essential_text_threshold=0.8,
):
    images = [Image.open(img_path).convert("RGB") for img_path in img_paths]

    tokenizer = processor.tokenizer

    # ── Token IDs ──────────────────────────────────────────────────────────────
    panel_id = tokenizer.convert_tokens_to_ids("<panel>")
    text_id = tokenizer.convert_tokens_to_ids("<text>")
    character_id = tokenizer.convert_tokens_to_ids("<character>")
    tail_id = tokenizer.convert_tokens_to_ids("<tail>")
    # category token → bucket key 的映射（text 单独处理，直接丢弃）
    token_map = {
        panel_id: "panel",
        character_id: "character",
        tail_id: "tail",
    }

    # ── Step 1: 原始前向推理，获取 panels / characters / tails ─────────────────
    batch_inputs = processor(
        batch_input_text=[
            "Find all panels, texts, characters, and speech-bubble tails in the image."
        ]
        * len(images),
        batch_input_list_of_list_of_bboxes=[[]] * len(images),
        batch_images=images,
        padding=True,
        truncation=True,
        max_input_length_including_image_tokens=1024,
        max_output_length=1024,
        return_tensors="pt",
        dtype=model.dtype,
        device=model.device,
    )

    generated_ids = model.generate(
        input_ids=batch_inputs["input_ids"],
        pixel_values=batch_inputs["pixel_values"],
        max_new_tokens=1024,
        do_sample=False,
        num_beams=3,
    )

    # ── Step 2: 解析原始检测结果（panels / characters / tails）────────────────
    (
        generated_texts,
        list_of_list_of_list_of_bboxes,
        batch_indices_of_bboxes_in_generated_text,
    ) = processor.postprocess_output(generated_ids, images)

    map_to_category = {
        "<pa": "panels",
        "<te": "texts",
        "<ch": "characters",
        "<ta": "tails",
    }

    results = []
    for generated_text, indices, list_of_bboxes in zip(
        generated_texts,
        batch_indices_of_bboxes_in_generated_text,
        list_of_list_of_list_of_bboxes,
    ):
        categories = [
            map_to_category.get(generated_text[j : j + 3], None) for _, j in indices
        ]
        result = {"panels": [], "texts": [], "characters": [], "tails": []}
        for category, bboxes in zip(categories, list_of_bboxes):
            if category is None:
                continue
            result[category].extend(bboxes)
        results.append(result)

    # ── Step 3: 构造注入外来 OCR boxes 后的 patched decoder 序列 ───────────────
    def encode_box_as_loc_tokens(box, img_w, img_h):
        """
        正确顺序: [loc_x1, loc_y1, loc_x2, loc_y2, text_id]
        category token 在最后，因为 <text> 的 hidden state
        通过因果注意力 attend 前面 4 个 loc token，
        从而将 box 坐标信息编码进去，影响后续 association head。
        """
        x1, y1, x2, y2 = box

        def norm(v, dim):
            return min(999, max(0, round(v / dim * 999)))

        locs = [norm(x1, img_w), norm(y1, img_h), norm(x2, img_w), norm(y2, img_h)]
        loc_ids = [tokenizer.convert_tokens_to_ids(f"<loc_{v}>") for v in locs]
        return loc_ids + [text_id]

    def parse_original_sequence(output_tokens):
        """
        正确切法: [loc, loc, loc, loc, category] 为一个 entry（共5个token）
        第 5 个 token 才是 category token。
        text entry 直接跳过（将被外来 OCR 替换）。
        """
        buckets = {"panel": [], "character": [], "tail": []}
        i = 1  # 跳过 BOS
        while i < len(output_tokens):
            # 尝试读取一个完整 entry（需要至少 5 个 token）
            if i + 4 < len(output_tokens):
                candidate_category = output_tokens[i + 4]
                if candidate_category in token_map:
                    # panel / character / tail：保留
                    entry = output_tokens[i : i + 5]
                    buckets[token_map[candidate_category]].append(entry)
                    i += 5
                    continue
                elif candidate_category == text_id:
                    # 原始 text entry：丢弃，由外来 OCR 替换
                    i += 5
                    continue
            i += 1
        return buckets

    # 先对 ocr_results 排序
    ordered_ocr_results = get_ordered_list(unordered_ocr_results, results)

    patched_decoder_ids = []

    for img_idx, image in enumerate(images):
        img_w, img_h = image.size
        gen_seq = generated_ids[img_idx]

        # 找到最后一个 BOS，取其后的 decoder 输出部分
        bos_positions = torch.where(gen_seq == tokenizer.bos_token_id)[0]
        bos_pos = bos_positions[-1].item()
        output_tokens = gen_seq[bos_pos:].tolist()  # [BOS, loc,loc,loc,loc,cat, ...]

        # 解析原始序列，提取 panel / character / tail entries
        buckets = parse_original_sequence(output_tokens)

        # 构造注入的 text entries
        ext = ordered_ocr_results[img_idx]
        injected_text_entries = [
            encode_box_as_loc_tokens(box, img_w, img_h) for box in ext["boxes"]
        ]

        # 重组序列：BOS + panels + injected_texts + characters + tails
        # 顺序与原始模型生成顺序一致
        new_seq = [output_tokens[0]]  # BOS
        for entry in buckets["panel"]:
            new_seq.extend(entry)  # [loc,loc,loc,loc,<panel>]
        for entry in injected_text_entries:
            new_seq.extend(entry)  # [loc,loc,loc,loc,<text>]  ← 外来 OCR
        for entry in buckets["character"]:
            new_seq.extend(entry)  # [loc,loc,loc,loc,<character>]
        for entry in buckets["tail"]:
            new_seq.extend(entry)  # [loc,loc,loc,loc,<tail>]

        patched_decoder_ids.append(
            torch.tensor(new_seq, dtype=torch.long, device=model.device)
        )

        # 用外来 OCR 的 boxes 和 texts 覆盖结果
        results[img_idx]["texts"] = [list(map(float, b)) for b in ext["boxes"]]
        if "texts" in ext:
            results[img_idx]["ocr_texts"] = ext["texts"]

    # 对齐 batch 长度
    patched_decoder_ids = pad_sequence(
        patched_decoder_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )

    # ── Step 4: 用 patched 序列重新跑 association heads ────────────────────────
    association_outputs = model(
        input_ids=batch_inputs["input_ids"],
        pixel_values=batch_inputs["pixel_values"],
        decoder_input_ids=patched_decoder_ids,
        tokenizer=tokenizer,
    )

    # ── Step 5: 提取 association 结果 ─────────────────────────────────────────
    for img_idx in range(len(results)):
        character_cluster_labels = UnionFind.from_adj_matrix(
            association_outputs.character_character_affinity_matrices[img_idx]
            > character_character_association_threshold
        ).get_labels_for_connected_components()

        text_character_association = torch.nonzero(
            association_outputs.text_character_association_matrices[img_idx]
            > text_character_association_threshold
        ).tolist()

        text_tail_association = torch.nonzero(
            association_outputs.text_tail_association_matrices[img_idx]
            > text_tail_association_threshold
        ).tolist()

        essential_text_logits = (
            association_outputs.essential_text_logits[img_idx]
            > essential_text_threshold
        ).tolist()

        results[img_idx]["character_cluster_labels"] = character_cluster_labels
        results[img_idx]["text_character_associations"] = text_character_association
        results[img_idx]["text_tail_associations"] = text_tail_association
        results[img_idx]["is_essential_text"] = essential_text_logits
        results[img_idx]["dialog_confidences"] = [1.0] * len(results[img_idx]["texts"])

    return results


import httpx


def get_ocr_results(image_paths: list[str], only_white_bg: bool, zh_texts: bool):
    url = "http://127.0.0.1:8000/ocr"

    params = {
        "image_paths": image_paths,
        "only_white_bg": only_white_bg,
        "zh_texts": zh_texts,
    }

    response = httpx.post(url, json=params, timeout=30.0)  # 设置合理的超时时间

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"OCR 服务调用失败: {response.text}")
