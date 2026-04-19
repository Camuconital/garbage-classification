from pathlib import Path

import gradio as gr
import torch
from PIL import Image
from torchvision import transforms

from model import build_model


DEFAULT_MODEL_PATH = Path('best_model.pth')
DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224


def load_checkpoint(model_path: Path):
    """
    加载训练得到的 checkpoint，并构建推理模型。
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f'未找到模型文件: {model_path}。请确认 best_model.pth 位于仓库根目录，'
            '或修改 gradio_app.py 中的 DEFAULT_MODEL_PATH。'
        )

    checkpoint = torch.load(model_path, map_location='cpu')
    class_names = checkpoint.get('class_names')
    mean = checkpoint.get('mean', DEFAULT_MEAN)
    std = checkpoint.get('std', DEFAULT_STD)

    if not class_names:
        raise ValueError('checkpoint 中未找到 class_names，无法输出类别名称。')

    model = build_model(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return model, class_names, transform


MODEL, CLASS_NAMES, TRANSFORM = load_checkpoint(DEFAULT_MODEL_PATH)


@torch.inference_mode()
def predict(image: Image.Image):
    """
    接收单张图片并返回 Top-3 预测结果。
    """
    if image is None:
        return '请先上传一张图片。', None

    image = image.convert('RGB')
    tensor = TRANSFORM(image).unsqueeze(0)
    logits = MODEL(tensor)
    probs = torch.softmax(logits, dim=1).squeeze(0)

    top_probs, top_indices = torch.topk(probs, k=min(3, len(CLASS_NAMES)))
    top_result = {
        CLASS_NAMES[idx]: float(prob)
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist())
    }

    best_idx = int(torch.argmax(probs).item())
    best_name = CLASS_NAMES[best_idx]
    best_prob = float(probs[best_idx].item())
    summary = f'预测类别：{best_name}（置信度：{best_prob:.2%}）'
    return summary, top_result


def build_demo():
    """
    构建 Gradio 页面。
    """
    with gr.Blocks(title='垃圾分类识别系统') as demo:
        gr.Markdown('## 垃圾分类识别系统（Gradio）')
        gr.Markdown('上传一张图片，模型会自动输出预测类别与各类别概率。')

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type='pil', label='上传图片')
                predict_btn = gr.Button('开始识别', variant='primary')
                clear_btn = gr.Button('清空')
            with gr.Column(scale=1):
                result_text = gr.Textbox(label='识别结果', interactive=False)
                result_label = gr.Label(label='Top-3 概率')

        predict_btn.click(
            fn=predict,
            inputs=image_input,
            outputs=[result_text, result_label],
        )
        clear_btn.click(
            fn=lambda: (None, '', None),
            inputs=None,
            outputs=[image_input, result_text, result_label],
        )

        gr.Examples(
            examples=[],
            inputs=image_input,
            label='你可以把常用测试图片路径添加到 gr.Examples 中。',
        )

    return demo


if __name__ == '__main__':
    demo = build_demo()
    demo.launch(server_name='0.0.0.0', server_port=7860)
