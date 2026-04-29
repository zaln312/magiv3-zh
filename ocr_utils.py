# 可视化tail
def visualize_tail(image, result):
    import numpy as np
    import matplotlib.pyplot as plt

    image_np = np.array(image)

    h, w = image_np.shape[:2]
    if h > w:
        fig, ax = plt.subplots(figsize=(8, 8 * h / w))
    else:
        fig, ax = plt.subplots(figsize=(8 * w / h, 8))

    ax.imshow(image_np)

    for i, bbox in enumerate(result["tails"]):
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        rect = plt.Rectangle(
            (x1, y1), width, height, linewidth=1, edgecolor="yellow", facecolor="none"
        )
        ax.add_patch(rect)

        # 可选：显示 index
        ax.text(x1, y1, str(i), color="yellow", fontsize=10)

    ax.axis("off")
    plt.show()


# 对 ocr box 按阅读顺序排序
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
        return unordered_ocr_result, []

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
    text_panel_associations = []  # 新增：每个排序后位置对应的 [原始索引, 面板索引]

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
                new_index = len(ordered_indices)  # box 排序后索引
                ordered_indices.append(item[0])
                text_panel_associations.append([new_index, pid])

    # ---------- build result ----------
    ordered_boxes = [boxes[i] for i in ordered_indices]
    ordered_texts = [texts[i] for i in ordered_indices]

    return {
        "img_path": img_path,
        "boxes": ordered_boxes,
        "texts": ordered_texts,
        "text_panel_associations": text_panel_associations,
    }


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

        results[img_idx]["text_panel_associations"] = ext["text_panel_associations"]

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


@torch.no_grad()
def predict_with_injected_ocr_and_global_id(
    model,
    processor,
    img_paths,
    unordered_ocr_results,
    character_character_association_threshold=0.5,
    text_character_association_threshold=0.8,
    text_tail_association_threshold=0.8,
    essential_text_threshold=0.8,
    global_id_threshold=0.8,
    global_character_library=None,
    debug=False,
):
    from PIL import Image
    from torch.nn.utils.rnn import pad_sequence
    from scipy.optimize import linear_sum_assignment

    if global_character_library is None:
        global_character_library = []

    images = [Image.open(p).convert("RGB") for p in img_paths]
    tokenizer = processor.tokenizer

    # ─────────────────────────────────────────────
    # Step 1: 原始 detection
    # ─────────────────────────────────────────────
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

    (
        generated_texts,
        list_of_list_of_list_of_bboxes,
        batch_indices,
    ) = processor.postprocess_output(generated_ids, images)

    map_to_category = {
        "<pa": "panels",
        "<te": "texts",
        "<ch": "characters",
        "<ta": "tails",
    }

    results = []
    for gen_text, indices, bboxes in zip(
        generated_texts, batch_indices, list_of_list_of_list_of_bboxes
    ):
        categories = [
            map_to_category.get(gen_text[j : j + 3], None) for _, j in indices
        ]
        r = {"panels": [], "texts": [], "characters": [], "tails": []}
        for c, bb in zip(categories, bboxes):
            if c:
                r[c].extend(bb)
        results.append(r)

    # ─────────────────────────────────────────────
    # Step 2: 构造 patched decoder ids（与你原来一致）
    # ─────────────────────────────────────────────
    def encode_box_as_loc_tokens(box, w, h):
        x1, y1, x2, y2 = box

        def norm(v, dim):
            return min(999, max(0, round(v / dim * 999)))

        locs = [norm(x1, w), norm(y1, h), norm(x2, w), norm(y2, h)]
        loc_ids = [tokenizer.convert_tokens_to_ids(f"<loc_{v}>") for v in locs]
        return loc_ids + [tokenizer.convert_tokens_to_ids("<text>")]

    def parse_original_sequence(tokens):
        panel_id = tokenizer.convert_tokens_to_ids("<panel>")
        character_id = tokenizer.convert_tokens_to_ids("<character>")
        tail_id = tokenizer.convert_tokens_to_ids("<tail>")
        text_id = tokenizer.convert_tokens_to_ids("<text>")

        token_map = {
            panel_id: "panel",
            character_id: "character",
            tail_id: "tail",
        }

        buckets = {"panel": [], "character": [], "tail": []}
        i = 1
        while i + 4 < len(tokens):
            cat = tokens[i + 4]
            if cat in token_map:
                buckets[token_map[cat]].append(tokens[i : i + 5])
                i += 5
            elif cat == text_id:
                i += 5
            else:
                i += 1
        return buckets

    ordered_ocr_results = get_ordered_list(unordered_ocr_results, results)

    patched_decoder_ids = []

    for img_idx, image in enumerate(images):
        w, h = image.size
        gen_seq = generated_ids[img_idx]

        bos = torch.where(gen_seq == tokenizer.bos_token_id)[0][-1].item()
        tokens = gen_seq[bos:].tolist()

        buckets = parse_original_sequence(tokens)

        ext = ordered_ocr_results[img_idx]
        injected = [encode_box_as_loc_tokens(b, w, h) for b in ext["boxes"]]

        new_seq = [tokens[0]]
        for e in buckets["panel"]:
            new_seq.extend(e)
        for e in injected:
            new_seq.extend(e)
        for e in buckets["character"]:
            new_seq.extend(e)
        for e in buckets["tail"]:
            new_seq.extend(e)

        patched_decoder_ids.append(torch.tensor(new_seq, device=model.device))

        # 覆盖 text
        results[img_idx]["texts"] = [list(map(float, b)) for b in ext["boxes"]]
        results[img_idx]["ocr_texts"] = ext.get("texts", [])
        results[img_idx]["text_panel_associations"] = ext.get(
            "text_panel_associations", []
        )

    patched_decoder_ids = pad_sequence(
        patched_decoder_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )

    # ─────────────────────────────────────────────
    # Step 3: 手动 forward（关键！拿 hidden states）
    # ─────────────────────────────────────────────
    image_features = model._encode_image(batch_inputs["pixel_values"])
    inputs_embeds, attention_mask = model._merge_input_ids_with_image_features(
        image_features, model.get_input_embeddings()(batch_inputs["input_ids"])
    )

    lm_outputs = model.language_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        decoder_input_ids=patched_decoder_ids,
        output_hidden_states=True,
        return_dict=True,
    )

    decoder_hidden = lm_outputs.decoder_hidden_states[-1]

    # ─────────────────────────────────────────────
    # Step 4: association heads（与你原逻辑一致）
    # ─────────────────────────────────────────────
    affinity = model.get_character_character_affinity_matrices(
        decoder_hidden, patched_decoder_ids, tokenizer, apply_sigmoid=True
    )
    text_char = model.get_text_character_association_matrices(
        decoder_hidden, patched_decoder_ids, tokenizer, apply_sigmoid=True
    )
    text_tail = model.get_text_tail_association_matrices(
        decoder_hidden, patched_decoder_ids, tokenizer, apply_sigmoid=True
    )
    essential = model.get_essential_text_logits(
        decoder_hidden, patched_decoder_ids, tokenizer, apply_sigmoid=True
    )

    # ⭐ 新增：character features
    char_features = model.extract_character_features(
        decoder_hidden, patched_decoder_ids, tokenizer
    )

    # ─────────────────────────────────────────────
    # Step 5: clustering + global ID
    # ─────────────────────────────────────────────
    for i in range(len(results)):

        cluster_labels = UnionFind.from_adj_matrix(
            affinity[i] > character_character_association_threshold
        ).get_labels_for_connected_components()

        results[i]["character_cluster_labels"] = cluster_labels
        results[i]["text_character_associations"] = torch.nonzero(
            text_char[i] > text_character_association_threshold
        ).tolist()
        results[i]["text_tail_associations"] = torch.nonzero(
            text_tail[i] > text_tail_association_threshold
        ).tolist()
        results[i]["is_essential_text"] = (
            essential[i] > essential_text_threshold
        ).tolist()

        feats = char_features[i]

        if feats.numel() == 0:
            results[i]["global_character_ids"] = []
            continue

        # ---- cluster 平均特征 ----
        cluster_ids = sorted(set(cluster_labels))
        cluster_feats = {}
        for cid in cluster_ids:
            idxs = [k for k, c in enumerate(cluster_labels) if c == cid]
            f = feats[idxs].mean(0)
            f = f / f.norm()
            cluster_feats[cid] = f

        # ---- Hungarian matching ----
        if not global_character_library:
            if debug:
                print(f"\n[DEBUG] Image {i}: global library empty → create new IDs")

            mapping = {}
            for cid, f in cluster_feats.items():
                gid = len(global_character_library)
                global_character_library.append(
                    {"global_id": gid, "features": f.clone()}
                )
                mapping[cid] = gid

                if debug:
                    print(f"  cluster {cid} → new global_id {gid}")

        else:
            lib_feats = torch.stack([e["features"] for e in global_character_library])
            lib_ids = [e["global_id"] for e in global_character_library]

            cluster_ids = list(cluster_feats.keys())
            cluster_stack = torch.stack([cluster_feats[c] for c in cluster_ids])

            sim = torch.mm(cluster_stack, lib_feats.t())

            if debug:
                print(f"\n{'='*60}")
                print(f"[DEBUG] Image {i}")
                print(f"clusters: {len(cluster_ids)} | library: {len(lib_ids)}")

                header = "cluster ".ljust(10) + "".join(
                    f"{gid:>10d}" for gid in lib_ids
                )
                print(header)
                for r_idx, cid in enumerate(cluster_ids):
                    row = f"{cid:<10}" + "".join(
                        f"{sim[r_idx, c].item():10.4f}" for c in range(len(lib_ids))
                    )
                    print(row)

            cost = -sim.cpu().numpy()
            r, c = linear_sum_assignment(cost)

            mapping = {}
            assigned = set()

            if debug:
                print("\n[DEBUG] Hungarian assignment:")

            for ri, ci in zip(r, c):
                cid = cluster_ids[ri]
                gid = lib_ids[ci]
                score = sim[ri, ci].item()

                if debug:
                    status = "✅ MATCH" if score >= global_id_threshold else "❌ NEW"
                    print(f"  cluster {cid} → global {gid} | sim={score:.4f} {status}")

                if score >= global_id_threshold:
                    mapping[cid] = gid
                    assigned.add(gid)

            # 未匹配 → 新建 ID
            for cid in cluster_ids:
                if cid not in mapping:
                    gid = len(global_character_library)
                    global_character_library.append(
                        {"global_id": gid, "features": cluster_feats[cid].clone()}
                    )
                    mapping[cid] = gid

                    if debug:
                        print(f"  cluster {cid} → NEW global_id {gid}")

            if debug:
                print(f"[DEBUG] final mapping: {mapping}")

        results[i]["global_character_ids"] = [mapping[c] for c in cluster_labels]

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


# def crop_panels(img_paths, results):
#     for img_path, result in zip(img_paths, results):
#         _crop(img_path, result["boxes"])


import io
import base64
from PIL import Image
from openai import OpenAI


def _get_caption(img: Image.Image, think: bool = False):
    """
    从单 panel 中获取描述
    """
    if img.mode != "RGB":
        img = img.convert("RGB")

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode()

    client = OpenAI(base_url="http://localhost:8001/v1", api_key="EMPTY")
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                },
                {
                    "type": "text",
                    # "text": "描述这张图片。重点关注角色、他们的外貌、他们的动作以及环境。忽略图片中的任何文字、对话或对话气泡。",
                    # "text": "Describe this image to me. Focus on the characters, their appearance, their actions, and the environment. Please ignore any text, dialogues, or speech bubbles in the image. Present the description in prose paragraph form, not as a list.",
                    "text": "Describe this image in a single prose paragraph. For each character, start by clearly stating their relative position (e.g., 'the character on the left', 'in the foreground', 'the girl on the right'), then describe their appearance (hair, clothing), and finally their actions or emotions. Do not use specific names. Ignore all embedded text, speech bubbles, and dialogue. Focus purely on visual elements.",
                },
            ],
        }
    ]
    if not think:
        response = client.chat.completions.create(
            model="Qwen3.5-4B",
            messages=messages,
            max_tokens=1024,
            temperature=0.7,
            top_p=0.8,
            presence_penalty=1.5,
            extra_body={
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
    else:
        response = client.chat.completions.create(
            model="Qwen3.5-4B",
            messages=messages,
            max_tokens=1024,
            temperature=1.0,
            top_p=0.95,
            presence_penalty=1.5,
            extra_body={
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": True},
            },
        )
    return response.choices[0].message.content


def get_captions(image_paths: list[str], results: list[dict], think: bool):
    assert len(image_paths) == len(results), "image_paths 和 results 长度不一致"

    captions_list = []
    for img_path, result in zip(image_paths, results):
        # 对于单张 img
        captions = []
        img = Image.open(img_path)
        for idx, panel in enumerate(result["panels"]):
            img_panel = img.crop(panel)
            img_panel.save(f"output/panel{idx}.jpg")
            caption = _get_caption(img_panel, think)
            captions.append(caption)
        captions_list.append(captions)
    return captions_list


from PIL import Image, ImageDraw, ImageFont
import numpy as np


def _get_center(box):
    """计算边界框的中心点 (x, y)"""
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _calculate_iou(box1, box2):
    """计算两个边界框的 IoU (交并比)"""
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    if inter_area == 0:
        return 0.0

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter_area / float(box1_area + box2_area - inter_area)


def preprocess_panel_characters(results_dict):
    """
    为 panel 分配 characters 并转换坐标
    输入: 单张图片的 results 字典，包含 'panels', 'characters', 'global_character_ids'
    输出: 一个列表，列表的每个元素对应一个 panel 的已归类 characters 信息
    """
    panels = results_dict["panels"]
    characters = results_dict["characters"]
    labels = results_dict["global_character_ids"]

    # 初始化每个 panel 拥有的 characters 列表
    panel_characters_list = [[] for _ in range(len(panels))]

    for char_idx, char_box in enumerate(characters):
        char_label = labels[char_idx]
        char_center = _get_center(char_box)

        best_panel_idx = -1
        min_dist = float("inf")

        # 遍历 panels，寻找重叠且中心点最近的
        for panel_idx, panel_box in enumerate(panels):
            # 简单判断是否有交集 (这里没除以面积，直接算交集面积即可判断重叠)
            xA = max(char_box[0], panel_box[0])
            yA = max(char_box[1], panel_box[1])
            xB = min(char_box[2], panel_box[2])
            yB = min(char_box[3], panel_box[3])
            inter_area = max(0, xB - xA) * max(0, yB - yA)

            if inter_area > 0:  # 如果存在重叠
                panel_center = _get_center(panel_box)
                # 计算中心点欧氏距离
                dist = np.linalg.norm(np.array(char_center) - np.array(panel_center))
                if dist < min_dist:
                    min_dist = dist
                    best_panel_idx = panel_idx

        # 归类到最佳 panel 并进行坐标系转换 (全局 -> 局部)
        if best_panel_idx != -1:
            target_panel = panels[best_panel_idx]
            local_box = [
                char_box[0] - target_panel[0],  # local x1
                char_box[1] - target_panel[1],  # local y1
                char_box[2] - target_panel[0],  # local x2
                char_box[3] - target_panel[1],  # local y2
            ]
            panel_characters_list[best_panel_idx].append(
                {"local_box": local_box, "cluster_label": char_label}
            )

    return panel_characters_list


def get_grounding(grounding_result, panel_characters, image=None, output_dir="output"):
    caption = grounding_result["grounded_caption"]
    bboxes = grounding_result["bboxes"]
    indices = grounding_result["indices_of_bboxes_in_caption"]

    # --- 逻辑 A: 通过局部 Box 匹配全局预设的 Cluster ID ---
    char_ids = []
    for bbox_list in bboxes:
        # grounding 返回的通常是嵌套列表，取第一个主框，它已经是基于当前 panel 的局部坐标了
        grounding_box = bbox_list[0]

        matched_id = (
            -1
        )  # 默认值：-1表示未匹配到任何预设角色 (比如 grounding 框住了一个苹果)
        best_iou = 0.0

        # 将 grounding 的框与该 panel 下已知的所有 character 框进行对比
        for p_char in panel_characters:
            iou = _calculate_iou(grounding_box, p_char["local_box"])
            if iou > best_iou:
                best_iou = iou
                matched_id = p_char["cluster_label"]

        # 设定一个宽容的 IoU 阈值 (比如 0.1)，因为检测模型和 Grounding 模型的框大小往往不完全一致
        if best_iou > 0.1:
            char_ids.append(matched_id)
        else:
            char_ids.append(-1)

    # --- 逻辑 B: 文本插入 ---
    # 根据 index 倒序排列，防止文本插入导致后续索引偏移
    indexed_items = sorted(zip(indices, char_ids), key=lambda x: x[0][0], reverse=True)
    tagged_caption = caption

    for (start, end), cid in indexed_items:
        # 只有匹配到了全局角色 (cid != -1)，才插入标签
        if cid != -1:
            tag = f"[{cid}]"
            tagged_caption = tagged_caption[:start] + tag + tagged_caption[start:]

    # --- 逻辑 C: 绘制 Bbox (修复坐标) ---
    if image is not None:
        draw_img = image.convert("RGB")
        draw = ImageDraw.Draw(draw_img)
        colors = ["#FF3333", "#33FF33", "#3333FF", "#FFFF33", "#FF33FF"]

        for idx, (box, cid) in enumerate(zip(bboxes, char_ids)):
            b = box[0]

            # 【核心修复】：直接使用原始值，不再乘以 width/1000
            x_min, y_min, x_max, y_max = (
                min(b[0], b[2]),
                min(b[1], b[3]),
                max(b[0], b[2]),
                max(b[1], b[3]),
            )

            color = colors[cid % len(colors)]
            draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)

            # 标签绘制
            label_y = y_min if y_min > 20 else y_max
            try:
                # 尝试画个背景，增加可读性
                draw.rectangle([x_min, label_y - 20, x_min + 25, label_y], fill=color)
                draw.text((x_min + 5, label_y - 18), str(cid), fill="white")
            except:
                pass

        # --- 新增：生成唯一文件名并保存 ---
        import os

        os.makedirs(output_dir, exist_ok=True)

        import uuid

        # 或者根据 caption 生成有意义的名字，这里简单使用时间戳
        import time

        filename = f"bbox_{int(time.time()*1000)}.jpg"
        save_path = os.path.join(output_dir, filename)
        draw_img.save(save_path, "JPEG")
        # print(f"保存图片到: {save_path}")
        # -----------------------------

    return tagged_caption


def build_panel_scripts(
    result_data,
    essential_only=False,
    include_narrator=True,
    label="旁白",
    label_char_name="角色",
):
    """
    为每个分镜构建剧本格式的对话串。

    参数:
        result_data: dict，对应 results[0] 的数据结构
        essential_only: bool，是否仅保留关键文本（is_essential_text 为 True）
        include_narrator: bool，是否保留无角色关联的文本（使用标注词 label）
        label: str，无角色关联的文本的标注词，默认 "旁白"


    返回:
        list: 每个元素为一个分镜的对话列表，格式为 "角色 X: “对话内容”"
    """
    panels = result_data["panels"]
    ocr_texts = result_data["ocr_texts"]
    text_panel_assoc = result_data["text_panel_associations"]
    text_char_assoc = result_data["text_character_associations"]
    char_cluster_labels = result_data["global_character_ids"]
    is_essential = result_data.get("is_essential_text", [False] * len(ocr_texts))

    # 建立 text_idx -> character_idx 的映射
    text_to_char = {t_idx: c_idx for t_idx, c_idx in text_char_assoc}

    # 按分镜收集文本索引（直接使用列表，保留原始顺序）
    panel_texts = {p_idx: [] for p_idx in range(len(panels))}
    for t_idx, p_idx in text_panel_assoc:
        panel_texts[p_idx].append(t_idx)

    all_panel_scripts = []

    for p_idx in range(len(panels)):
        texts_in_panel = panel_texts.get(p_idx, [])
        if not texts_in_panel:
            all_panel_scripts.append([])
            continue

        panel_script = []
        for t_idx in texts_in_panel:  # 已按阅读顺序排列
            # 筛选 essential
            if essential_only and not is_essential[t_idx]:
                continue

            text_content = ocr_texts[t_idx]

            if t_idx in text_to_char:
                char_idx = text_to_char[t_idx]
                cluster_id = char_cluster_labels[char_idx]
                line = f"{label_char_name} {cluster_id}: “{text_content}”"
                panel_script.append(line)
            elif include_narrator:
                # 无角色关联，作为旁白处理
                line = f"{label}: “{text_content}”"
                panel_script.append(line)
            # 否则忽略该文本

        all_panel_scripts.append(panel_script)

    return all_panel_scripts


# @torch.no_grad()
# def predict_character_associations_only(
#     model,
#     processor,
#     images,
#     character_character_association_threshold=0.5,
# ):
#     # images = [Image.open(img_path).convert("RGB") for img_path in img_paths]

#     # 1. 仅请求检测角色，减少生成的 token 长度
#     batch_inputs = processor(
#         batch_input_text=["Find all characters in the image."] * len(images),
#         batch_input_list_of_list_of_bboxes=[[]] * len(images),
#         batch_images=images,
#         padding=True,
#         truncation=True,
#         max_input_length_including_image_tokens=1024,
#         max_output_length=512,  # 仅检测角色，通常不需要 1024 那么长
#         return_tensors="pt",
#         dtype=model.dtype,
#         device=model.device,
#     )

#     # 2. 生成检测序列
#     generated_ids = model.generate(
#         input_ids=batch_inputs["input_ids"],
#         pixel_values=batch_inputs["pixel_values"],
#         max_new_tokens=512,
#         do_sample=False,
#         num_beams=3,
#     )

#     # 3. 后处理得到检测框
#     (
#         generated_texts,
#         list_of_list_of_list_of_bboxes,
#         batch_indices_of_bboxes_in_generated_text,
#     ) = processor.postprocess_output(generated_ids, images)

#     results = []

#     # 4. 提取 Character 框
#     # 由于 Prompt 变了，理论上生成的几乎全是角色，但为了鲁棒性仍保留过滤逻辑
#     for gen_text, indices, bboxes in zip(
#         generated_texts,
#         batch_indices_of_bboxes_in_generated_text,
#         list_of_list_of_list_of_bboxes,
#     ):
#         # 修正点：提取 bboxes[i][0] 而不是整个 bboxes[i]
#         char_bboxes = [
#             (
#                 bboxes[i][0]
#                 if (isinstance(bboxes[i], list) and len(bboxes[i]) > 0)
#                 else bboxes[i]
#             )
#             for i, (start, end) in enumerate(indices)
#             if gen_text[start : start + 3] == "<ch"
#         ]
#         results.append({"characters": char_bboxes})

#     # 5. 准备 Decoder 输入用于计算关联矩阵
#     cleaned_generated_ids = []
#     for generated_id in generated_ids:
#         # 找到最后一个 BOS token 之后的内容
#         bos_indices = torch.where(generated_id == processor.tokenizer.bos_token_id)[0]
#         index_of_last_bos = bos_indices[-1].item() if len(bos_indices) > 0 else 0
#         cleaned_generated_ids.append(generated_id[index_of_last_bos:])

#     cleaned_generated_ids = pad_sequence(
#         cleaned_generated_ids,
#         batch_first=True,
#         padding_value=processor.tokenizer.pad_token_id,
#     )

#     # 6. 前向传播获取关联矩阵
#     association_outputs = model(
#         input_ids=batch_inputs["input_ids"],
#         pixel_values=batch_inputs["pixel_values"],
#         decoder_input_ids=cleaned_generated_ids,
#         tokenizer=processor.tokenizer,
#     )

#     # 7. 仅计算角色聚类标签
#     for img_idx in range(len(results)):
#         affinity_matrix = association_outputs.character_character_affinity_matrices[
#             img_idx
#         ]
#         cluster_labels = UnionFind.from_adj_matrix(
#             affinity_matrix > character_character_association_threshold
#         ).get_labels_for_connected_components()
#         results[img_idx]["character_cluster_labels"] = cluster_labels

#     return results


# @torch.no_grad()
# def predict_with_injected_characters(
#     model,
#     processor,
#     images,
#     injected_characters,  # 格式: [[ [x1,y1,x2,y2], ... ], [ [x1,y1,x2,y2], ... ]] 每张图一个列表
#     character_character_association_threshold=0.5,
# ):
#     tokenizer = processor.tokenizer
#     device = model.device

#     # 1. 准备基础 Input (Prompt)
#     # 即使我们注入框，通常也需要一个 prompt 来引导模型进入正确的任务状态
#     batch_inputs = processor(
#         batch_input_text=["Find all characters in the image."] * len(images),
#         batch_input_list_of_list_of_bboxes=[[]] * len(images),
#         batch_images=images,
#         padding=True,
#         truncation=True,
#         max_input_length_including_image_tokens=1024,
#         return_tensors="pt",
#         dtype=model.dtype,
#         device=device,
#     )

#     # 2. 辅助函数：将坐标转换为模型的 <loc_n> token 格式
#     def encode_box_as_tokens(box, img_w, img_h):
#         x1, y1, x2, y2 = box

#         def norm(v, dim):
#             return min(999, max(0, round(v / dim * 999)))

#         locs = [norm(x1, img_w), norm(y1, img_h), norm(x2, img_w), norm(y2, img_h)]
#         loc_ids = [tokenizer.convert_tokens_to_ids(f"<loc_{v}>") for v in locs]
#         # 后面紧跟 <character> 类别 token
#         char_token_id = tokenizer.convert_tokens_to_ids("<character>")
#         return loc_ids + [char_token_id]

#     # 3. 手动构造 Decoder Input 序列
#     patched_decoder_ids = []
#     bos_token_id = tokenizer.bos_token_id

#     for img_idx, image in enumerate(images):
#         img_w, img_h = image.size
#         # 起始符
#         new_seq = [bos_token_id]

#         # 注入 character 框
#         current_image_chars = injected_characters[img_idx]
#         for box in current_image_chars:
#             entry = encode_box_as_tokens(box, img_w, img_h)
#             new_seq.extend(entry)

#         patched_decoder_ids.append(
#             torch.tensor(new_seq, dtype=torch.long, device=device)
#         )

#     # 对齐 Batch 长度
#     patched_decoder_ids = pad_sequence(
#         patched_decoder_ids,
#         batch_first=True,
#         padding_value=tokenizer.pad_token_id,
#     )

#     # 4. 前向传播获取关联矩阵
#     # 此时 model 不再 generate，而是根据我们提供的序列计算 hidden states
#     association_outputs = model(
#         input_ids=batch_inputs["input_ids"],
#         pixel_values=batch_inputs["pixel_values"],
#         decoder_input_ids=patched_decoder_ids,
#         tokenizer=tokenizer,
#     )

#     # 5. 后处理结果
#     results = []
#     for img_idx in range(len(images)):
#         # 获取该图对应的 character-character 亲和力矩阵
#         affinity_matrix = association_outputs.character_character_affinity_matrices[
#             img_idx
#         ]

#         # 使用并查集 (UnionFind) 进行聚类
#         cluster_labels = UnionFind.from_adj_matrix(
#             affinity_matrix > character_character_association_threshold
#         ).get_labels_for_connected_components()

#         results.append(
#             {
#                 "characters": injected_characters[img_idx],
#                 "character_cluster_labels": cluster_labels,
#             }
#         )

#     return results


import os
import numpy as np
from PIL import Image, ImageDraw, ImageColor, ImageFont


def visualize_character_associations(images, results, output_dir="output"):
    """
    可视化角色检测框及其聚类结果
    :param images: 图像对象列表
    :param results: predict_character_associations_only 返回的结果列表
    :param output_dir: 保存目录
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 预定义一组高对比度颜色
    color_palette = [
        "#E00000",
        "#00CE00",
        "#0000FF",
        "#DBDB06",
        "#DD00DD",
        "#00E0E0",
        "#FFA500",
        "#800080",
        "#008000",
        "#000080",
        "#A52A2A",
        "#D8A4AD",
    ]

    for img_idx, (img, res) in enumerate(zip(images, results)):
        # 1. 加载图片
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("arialbd.ttf", 20)  # 字号 16
        except IOError:
            font = ImageFont.load_default()

        characters = res.get("characters", [])
        labels = res.get("global_character_ids", [])

        # 2. 绘制每一个角色框
        for i, (bbox, label) in enumerate(zip(characters, labels)):
            # 获取颜色（如果 label 超过预设颜色数则取模）
            color = color_palette[label % len(color_palette)]
            xmin, ymin, xmax, ymax = bbox

            # 绘制矩形框，设置宽度为 3-5 像素以便看清
            draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=4)

            # 在框上方绘制 Cluster ID 标签
            label_text = f"{label}"
            # 简单的文本背景，提升可读性
            draw.rectangle([xmin, ymin, xmin + 25, ymin + 20], fill=color)
            draw.text((xmin + 3, ymin), label_text, fill="white", font=font)

        # 3. 保存图片
        save_path = os.path.join(output_dir, f"{img_idx}_characters.jpg")
        img.save(save_path)
        print(f"Visual saved: {save_path}")


def get_prose_prompt(grounded_captions, panel_scripts):
    prose_prompt = []
    prose_prompt.append("I have a series of manga panel descriptions and dialogues.")
    prose_prompt.append("")
    global_panel_count = 1

    for img_idx, (captions, scripts) in enumerate(
        zip(grounded_captions, panel_scripts)
    ):
        # prose_prompt.append(f"Image {img_idx+1}")
        for panel_idx, (caption, script) in enumerate(zip(captions, scripts)):
            # prose_prompt.append(f"Panel {panel_idx+1}")
            prose_prompt.append(f"Panel {global_panel_count}")
            prose_prompt.append("")
            global_panel_count += 1

            prose_prompt.append("Description:")
            prose_prompt.append(f"{caption}")

            prose_prompt.append("Dialogues: ")
            for line in script:
                prose_prompt.append(f"{line}")

            prose_prompt.append("")

    prose_prompt.append(
        "I want you to write a summary in Chinese so that a blind or visually impaired person can understand the story. Make sure to stick to the provided details. All these panels belong to the same page so make sure your narrative is coherent. The format of the narrative should be a prose."
    )
    return prose_prompt
