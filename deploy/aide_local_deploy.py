import argparse
from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from data.dct import DCT_base_Rec_Module
from models.AIDE import AIDE


class AIDEPredictor:
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        self.model = AIDE(resnet_path=None, convnext_path=None).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        state_dict = checkpoint.get("model", checkpoint)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.resize = transforms.Resize([256, 256])
        self.dct = DCT_base_Rec_Module()

    @torch.no_grad()
    def predict(self, image: Image.Image):
        image = image.convert("RGB")
        image_tensor = self.to_tensor(image)

        x_minmin, x_maxmax, x_minmin1, x_maxmax1 = self.dct(image_tensor)
        x_0 = self.normalize(self.resize(image_tensor))
        x_minmin = self.normalize(self.resize(x_minmin))
        x_maxmax = self.normalize(self.resize(x_maxmax))
        x_minmin1 = self.normalize(self.resize(x_minmin1))
        x_maxmax1 = self.normalize(self.resize(x_maxmax1))

        x = torch.stack([x_minmin, x_maxmax, x_minmin1, x_maxmax1, x_0], dim=0).unsqueeze(0)
        x = x.to(self.device, non_blocking=True)

        logits = self.model(x)
        probs = torch.softmax(logits, dim=-1)[0]
        p_real = float(probs[0].item())
        p_fake = float(probs[1].item())
        return p_real, p_fake


@st.cache_resource(show_spinner=True)
def load_predictor(ckpt_path: str, device: str):
    return AIDEPredictor(ckpt_path, device)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args, _ = parser.parse_known_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        st.error(f"Checkpoint not found: {ckpt_path}")
        st.stop()

    st.set_page_config(page_title="AIDE Local Deployment", layout="centered")
    st.title("AIDE 本地部署（单图推理）")
    st.caption("输出‘AI生成概率’变量：P(fake)")

    predictor = load_predictor(str(ckpt_path), args.device)

    file = st.file_uploader("上传图片（jpg/png/webp）", type=["jpg", "jpeg", "png", "webp"])
    if file is None:
        st.info("请先上传一张图片。")
        return

    image = Image.open(file).convert("RGB")
    st.image(image, caption="输入图片", use_column_width=True)

    if st.button("开始推理", type="primary"):
        with st.spinner("推理中..."):
            p_real, p_fake = predictor.predict(image)

        st.metric("AI生成概率 P(fake)", f"{p_fake:.4f}")
        st.metric("真实图片概率 P(real)", f"{p_real:.4f}")
        st.progress(min(max(p_fake, 0.0), 1.0))

        if p_fake >= 0.5:
            st.success("判定：更可能是 AI 生成图片")
        else:
            st.info("判定：更可能是真实图片")


if __name__ == "__main__":
    main()
