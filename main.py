img_paths = [
    "/home/zaln/文档/AAvscode/magiv3/input/0007.jpg",
    "/home/zaln/文档/AAvscode/magiv3/input/0008.jpg",
]




# ---------- 启动 paddleocr ----------
conda activate paddleocr
uvicorn ocr_server:app


# ocr 流程
global_character_library = []

from ocr_utils import get_ocr_results, prepare_ordered_ocr_and_detect, predict_with_injected_ocr_and_global_id
from model.florence2.utils import visualise_single_image_prediction

unordered_ocr_res = get_ocr_results(img_paths, only_white_bg=False, zh_texts=True)
# ---------- 关闭 paddleocr ----------

# ---------- 启动 magiv3 ----------
from transformers import AutoProcessor, AutoModelForCausalLM
import torch


model_path = "model/florence2"
model = (
    AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        # load_in_4bit=True,
        trust_remote_code=True,
    )
    .cuda()
    .eval()
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

# 前半部分：检测 + 排序（包含在 ocr 流程中）
images, batch_inputs, generated_ids, results, ordered_ocr_results = prepare_ordered_ocr_and_detect(
    model,
    processor,
    img_paths,
    unordered_ocr_res,
)

# predict
results = predict_with_injected_ocr_and_global_id(
    model,
    processor,
    images,
    batch_inputs,
    generated_ids,
    results,
    ordered_ocr_results,
    global_character_library=global_character_library,
    debug=True,
)

# ---------- 关闭 magiv3 ----------
del model
del processor
torch.cuda.empty_cache()




# debug 绘制
import numpy as np
from PIL import Image
from ocr_utils import print_texts


images = [Image.open(img).convert("RGB") for img in img_paths]
for result in results:
    result["dialog_confidences"] = [1.0] * len(result["texts"])


for img_idx in range(len(images)):
    visualise_single_image_prediction(
        np.array(images[img_idx]),
        results[img_idx],
        "output/" + "Overall_" + img_paths[img_idx].split("/")[-1],
    )
    print_texts(img_paths[img_idx], results[img_idx]["ocr_texts"])

from ocr_utils import visualize_character_associations

visualize_character_associations(images, results, "./output")

# ---------- 启动 llamacpp ----------
conda activate llamacpp
cd ~/文档/AAvscode/llama.cpp
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
./build/bin/llama-server \
    -m ./models/Qwen3.5-9B-Q6_K.gguf \
    --mmproj ./models/mmproj-F16-9B.gguf \
    --ctx-size 16384 \
    --port 8001 \
    --host 127.0.0.1

# caption
from ocr_utils import get_captions

captions = get_captions(img_paths, results, think=False)

# ---------- 关闭 llamacpp ----------

# ---------- 启动 magiv3 ----------
model = (
    AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    .cuda()
    .eval()
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

# character grounding
from PIL import Image

grounded_results = []
for i in range(len(img_paths)):
    grounded_result = []
    img = Image.open(img_paths[i])
    panel_imgs = [img.crop(panel) for panel in results[i]["panels"]]
    for panel_img, cap in zip(panel_imgs, captions[i]):
        res = model.predict_character_grounding(
            [panel_img],
            [cap],
            processor,
            strict=False,
        )
        grounded_result.extend(res)
    grounded_results.append(grounded_result)

# ---------- 关闭 magiv3 ----------


# caption 插入 [id]
from ocr_utils import preprocess_panel_characters, get_grounding

grounded_captions = []
for img_idx in range(len(img_paths)):
    panel_characters_list = preprocess_panel_characters(results[img_idx])
    grounded_result = grounded_results[img_idx]
    grounded_caption = []
    for panel_idx, res in enumerate(grounded_result):
        panel_characters = panel_characters_list[panel_idx]

        # 无可视化
        cap = get_grounding(res, panel_characters, None)

        # # 可视化
        # panel_img = (
        #     Image.open(img_paths[img_idx])
        #     .crop(results[img_idx]["panels"][panel_idx])
        #     .convert("RGB")
        # )
        # cap = get_grounding(res, panel_characters, panel_img)

        grounded_caption.append(cap)
    grounded_captions.append(grounded_caption)

# panel_scripts
from ocr_utils import build_panel_scripts

panel_scripts = []
for result in results:
    panel_scripts.append(
        build_panel_scripts(
            result,
            essential_only=False,
            include_narrator=True,
            label="narrator",
            label_char_name="character",
        )
    )
# prose_prompt
from ocr_utils import get_prose_prompt

prose_prompt = get_prose_prompt(grounded_captions, panel_scripts)

# prose
from ocr_utils import get_prose

prose = get_prose(prose_prompt)
