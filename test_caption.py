import sys
import io
import base64
from pathlib import Path
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
                    "text": "Describe this image in a single prose paragraph. For each character, start by clearly stating their relative position (e.g., 'the character on the left', 'in the foreground', 'the girl on the right'), then describe their appearance (hair, clothing), and finally their actions or emotions. Do not use specific names. Ignore all embedded text, speech bubbles, and dialogue. Focus purely on visual elements.",
                },
            ],
        }
    ]
    if not think:
        response = client.chat.completions.create(
            model="Qwen3",
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
            model="Qwen3",
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


def main():
    print("=" * 60)
    print("测试 _get_caption 方法")
    print("=" * 60)

    # if len(sys.argv) > 1:
    #     image_path = sys.argv[1]
    # else:
    #     image_path = input("请输入测试图片路径: ").strip()
    image_path = "/home/zaln/文档/AAvscode/magiv3/backend/output/panel5.jpg"

    if not Path(image_path).exists():
        print(f"错误: 文件不存在 - {image_path}")
        sys.exit(1)

    print(f"\n加载图片: {image_path}")
    img = Image.open(image_path)
    print(f"图片尺寸: {img.size}, 模式: {img.mode}")

    print("\n" + "-" * 60)
    print("测试 1: 非思考模式 (think=False)")
    print("-" * 60)
    try:
        caption_no_think = _get_caption(img, think=False)
        print("\n生成的 Caption:")
        print(caption_no_think)
    except Exception as e:
        print(f"\n错误: {e}")

    # print("\n" + "-" * 60)
    # print("测试 2: 思考模式 (think=True)")
    # print("-" * 60)
    # try:
    #     caption_think = _get_caption(img, think=True)
    #     print("\n生成的 Caption:")
    #     print(caption_think)
    # except Exception as e:
    #     print(f"\n错误: {e}")

    # print("\n" + "=" * 60)
    # print("测试完成")
    # print("=" * 60)


if __name__ == "__main__":
    main()
