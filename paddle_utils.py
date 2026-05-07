from opencc import OpenCC

# ==================== 背景筛选 ====================
def _is_white(
    image, poly, white_v_thresh=200, white_s_thresh=30, white_ratio_thresh=0.6
):
    """判断poly边框是否主要为白色"""
    h, w = image.shape[:2]
    pts = np.array(poly, dtype=np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.polylines(mask, [pts], isClosed=True, color=255, thickness=1)

    border_pixels = image[mask == 255]
    if border_pixels.size == 0:
        return False

    # BGR -> HSV
    hsv = cv2.cvtColor(border_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(
        -1, 3
    )

    white_mask = (hsv[:, 2] >= white_v_thresh) & (hsv[:, 1] <= white_s_thresh)

    return white_mask.mean() >= white_ratio_thresh


def filter_white_bg(data):
    """过滤白色背景的文本框"""
    img_path = data["img_path"]
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"无法读取图片: {img_path}")

    keep_polys, keep_texts = [], []
    for poly, text in zip(data["polys"], data["texts"]):
        if _is_white(image, poly):
            keep_polys.append(poly)
            keep_texts.append(text)

    return {"img_path": img_path, "polys": keep_polys, "texts": keep_texts}


# ==================== 文本处理 ====================
cc = OpenCC("t2s")  # 繁2简


def _clean_text(text):
    """清理文本：去空格、标点转换、繁转简"""
    if not text:
        return None

    text = text.strip()
    if text == "":
        return None

    # 标点转换
    punct_map = {
        ".": "。",
        "°": "。",
        ",": "，",
        ":": "：",
        ";": "；",
        "!": "！",
        "?": "？",
    }
    for en, zh in punct_map.items():
        text = text.replace(en, zh)

    return cc.convert(text)


def filter_texts(data):
    """过滤无效文本"""
    assert len(data["polys"]) == len(data["texts"]), "polys 与 texts 数量不一致"

    new_polys, new_texts = [], []
    for poly, text in zip(data["polys"], data["texts"]):
        cleaned = _clean_text(text)
        if cleaned:
            new_polys.append(poly)
            new_texts.append(cleaned)

    return {"img_path": data["img_path"], "polys": new_polys, "texts": new_texts}


# ==================== 合并靠近框 ====================
def _poly_to_box(poly):
    """多边形转边界框 [x1,y1,x2,y2]"""
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return [min(xs), min(ys), max(xs), max(ys)]


def _boxes_close(b1, b2, thresh=10):
    """判断两个 box 是否靠近"""
    dx = max(0, max(b1[0] - b2[2], b2[0] - b1[2]))
    dy = max(0, max(b1[1] - b2[3], b2[1] - b1[3]))
    return max(dx, dy) <= thresh


def _group_polys(polys):
    """靠近的 poly 聚合成组"""
    n = len(polys)
    bboxes = [_poly_to_box(p) for p in polys]
    visited = [False] * n

    groups = []

    for i in range(n):
        if visited[i]:
            continue

        queue = [i]
        visited[i] = True
        group = [i]

        while queue:
            cur = queue.pop(0)
            for j in range(n):
                if visited[j]:
                    continue
                if _boxes_close(bboxes[cur], bboxes[j]):
                    visited[j] = True
                    queue.append(j)
                    group.append(j)

        groups.append(group)

    return groups


def _merge_group_2_box(data, group):
    """
    组内 poly 排序、合并

    - input: one poly group
    - return: box [x1, y1, x2, y2]
    """
    items = []
    for idx in group:
        poly = data["polys"][idx]
        text = data["texts"][idx]
        x1, y1, x2, y2 = _poly_to_box(poly)
        items.append(((x1 + x2) / 2, (y1 + y2) / 2, poly, text))

    # 气泡内文字阅读顺序
    # 排序：右 -> 左，上 -> 下
    items.sort(key=lambda x: (-x[0], x[1]))

    merged_text = "".join(item[3] for item in items)

    # 合并 poly 坐标
    all_x = []
    all_y = []
    for _, _, poly, _ in items:
        for x, y in poly:
            all_x.append(x)
            all_y.append(y)

    merged_box = [min(all_x), min(all_y), max(all_x), max(all_y)]

    return merged_box, merged_text


def merge(data):
    """
    - input: dict(img_path, polys, texts)
    - return: dict(img_path, boxes, texts)
        box: [x1, y1, x2, y2]
    """
    merged_boxes = []
    merged_texts = []

    for group in _group_polys(data["polys"]):
        box, text = _merge_group_2_box(data, group)
        merged_boxes.append(box)
        merged_texts.append(text)

    assert len(merged_boxes) == len(merged_texts), "boxes 与 texts 数量不一致"

    return {
        "img_path": data["img_path"],
        "boxes": merged_boxes,
        "texts": merged_texts,
    }


# ==================== 可视化 ====================
def draw_data(data, output_dir="output"):
    """绘制结果并保存"""
    img = cv2.imread(data["img_path"])
    if img is None:
        return

    h, w = img.shape[:2]

    print(f"\n图片: {data['img_path']}")
    for i, text in enumerate(data["texts"], 1):
        print(f"[{i}] {text}")

    for i, box in enumerate(data["boxes"], 1):
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        # 画框
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        # 标序号
        cv2.putText(
            img,
            str(i),
            (x1, max(y1 - 5, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, os.path.basename(data["img_path"]))
    cv2.imwrite(out_path, img)
    print(f"\n结果已保存: {out_path}")
