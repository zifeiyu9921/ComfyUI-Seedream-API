"""ComfyUI adapter for the supplied Seedream task API."""

import json
import logging
import math
import os
import re
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

import folder_paths
from comfy.model_management import throw_exception_if_processing_interrupted
from comfy.utils import ProgressBar
from nodes import SaveImage

from .seedream_client import ASPECT_RATIOS, SeedreamClient, require_api_key


def read_config(api_key_override=""):
    api_key = require_api_key(api_key_override)
    path = Path(__file__).with_name("config.json")
    config = json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}
    base_url = os.environ.get("SEEDREAM_BASE_URL") or config.get("base_url")
    timeout = float(config.get("timeout", 3000))
    poll_interval = float(config.get("poll_interval", 2))
    if not math.isfinite(timeout) or not 1 <= timeout <= 86400:
        raise ValueError("config.json 的 timeout 必须在 1–86400 秒之间。")
    if not math.isfinite(poll_interval) or not 0.1 <= poll_interval <= 30:
        raise ValueError("config.json 的 poll_interval 必须在 0.1–30 秒之间。")
    return base_url, api_key, timeout, poll_interval


class SeedreamAPIGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "保留参考图主体，优化光影与细节，画面清晰自然。",
                    "tooltip": "自定义提示词：描述需要生成或修改的画面。",
                }),
                "ratio": (["auto", *ASPECT_RATIOS], {
                    "default": "auto",
                    "tooltip": "输出比例。auto 根据第一张参考图匹配最接近的支持比例。",
                }),
                "api_key": ("STRING", {"default": "", "dynamicPrompts": False,
                    "tooltip": "必填：输入有效的云端 API 密钥。留空不能运行。点击眼睛显示或隐藏。"}),
            },
            "optional": {"image": ("IMAGE", {
                "tooltip": "连接加载图像节点；不连接时使用纯文字生图。",
            })},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("生成图片",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "generate"
    CATEGORY = "Seedream API"
    DESCRIPTION = "上传参考图、填写提示词和比例，通过 Seedream API 生成图片。每次运行提交一个新任务。"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # This API exposes no seed: pressing Run must create a fresh task.
        return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, api_key=""):
        try:
            require_api_key(api_key)
        except ValueError as error:
            return str(error)
        return True

    def generate(self, prompt, ratio, image=None, api_key="", reference_images=()):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("请先填写提示词。")
        base_url, api_key, timeout, poll_interval = read_config(api_key)
        throw_exception_if_processing_interrupted()
        progress = ProgressBar(100)
        progress.update_absolute(0)
        last_progress = None

        def on_progress(task):
            nonlocal last_progress
            throw_exception_if_processing_interrupted()
            raw = task.get("progress")
            message = raw.get("message", raw) if isinstance(raw, dict) else raw
            percent = raw.get("percent") if isinstance(raw, dict) else None
            if message is not None and str(message) != last_progress:
                last_progress = str(message)
                safe = last_progress.replace(api_key, "***").replace("\n", " ")[:250]
                logging.info("[Seedream API] %s", safe)
            if isinstance(raw, str) and re.fullmatch(r"\s*\d+(?:\.\d+)?\s*%\s*", raw):
                percent = float(raw.strip().rstrip("%"))
            if isinstance(percent, (int, float)) and math.isfinite(percent):
                progress.update_absolute(max(0, min(99, int(percent))))

        temp_root = Path(folder_paths.get_temp_directory())
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="seedream_", dir=temp_root) as temp:
            references = []
            batches = ([image] if image is not None else []) + list(reference_images)
            for batch in batches:
                if batch.ndim != 4 or batch.shape[0] < 1 or batch.shape[-1] not in (1, 3, 4):
                    raise ValueError("参考图必须是 ComfyUI 的 IMAGE 图片输入。")
                for frame in batch:
                    throw_exception_if_processing_interrupted()
                    pixels = np.clip(frame.detach().to(device="cpu", dtype=torch.float32).numpy() * 255, 0, 255).round().astype(np.uint8)
                    if pixels.shape[-1] == 1:
                        pixels = pixels[..., 0]
                    path = Path(temp) / f"reference_{len(references) + 1}.png"
                    Image.fromarray(pixels).save(path)
                    references.append(path)
            client = SeedreamClient(base_url=base_url, api_key=api_key)
            try:
                paths = client.generate(
                    prompt=prompt.strip(), images=references, ratio=ratio,
                    output_dir=Path(temp) / "results", timeout=timeout,
                    poll_interval=poll_interval, on_progress=on_progress,
                    check_cancel=throw_exception_if_processing_interrupted,
                )
            finally:
                client.close()
            outputs = []
            for path in paths:
                throw_exception_if_processing_interrupted()
                with Image.open(path) as result:
                    rgb = ImageOps.exif_transpose(result).convert("RGB")
                    outputs.append(torch.from_numpy(np.array(rgb, dtype=np.float32) / 255.0).unsqueeze(0))
        progress.update_absolute(100)
        # A list preserves each output's original dimensions, including mixed sizes.
        return (outputs,)


class SeedreamAPIGenerateV2(SeedreamAPIGenerate):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"forceInput": True, "tooltip": "连接独立的文本节点。"}),
                "ratio": (["auto", *ASPECT_RATIOS], {"default": "auto", "tooltip": "输出图片比例；auto 跟随第一张参考图。"}),
                "api_key": ("STRING", {"default": "", "dynamicPrompts": False,
                    "tooltip": "必填：输入有效的云端 API 密钥。默认隐藏，点击眼睛显示；留空不能运行。"}),
            },
            "optional": {f"image{i}": ("IMAGE", {"tooltip": f"第 {i} 张参考图，可不连接。"}) for i in range(1, 11)},
        }

    def generate(self, prompt, ratio, api_key="", **images):
        references = [images[f"image{i}"] for i in range(1, 11) if images.get(f"image{i}") is not None]
        return super().generate(prompt, ratio, api_key=api_key, reference_images=references)


class SeedreamPromptText:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text": ("STRING", {"multiline": True, "dynamicPrompts": False,
                "default": "保留参考图主体，优化光影与细节，画面清晰自然。"})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("文本",)
    FUNCTION = "get_text"
    CATEGORY = "Seedream API"

    def get_text(self, text):
        return (text,)


class SeedreamLongEdge:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("INT", {"default": 1536, "min": 64, "max": 8192, "step": 1})}}

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("最大边长",)
    FUNCTION = "get_value"
    CATEGORY = "Seedream API"

    def get_value(self, value):
        return (value,)


class SeedreamResizeLongEdge:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "long_edge": ("INT", {"default": 1536, "forceInput": True, "min": 64, "max": 8192}),
            "method": (["lanczos", "bicubic", "bilinear"], {"default": "lanczos"}),
        }}

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("图像", "宽度", "高度")
    FUNCTION = "resize"
    CATEGORY = "Seedream API"
    DESCRIPTION = "按原比例缩小上传参考图，只缩小、不放大，不改变服务端输出分辨率。"

    def resize(self, image, long_edge, method):
        if not isinstance(long_edge, int) or not 64 <= long_edge <= 8192:
            raise ValueError("最大边长必须在 64–8192 之间。")
        if method not in {"lanczos", "bicubic", "bilinear"}:
            raise ValueError("不支持的缩放方式。")
        height, width = image.shape[1:3]
        scale = min(1.0, long_edge / max(height, width))
        new_width, new_height = max(1, round(width * scale)), max(1, round(height * scale))
        if (new_width, new_height) == (width, height):
            return image, width, height
        frames = []
        for frame in image:
            throw_exception_if_processing_interrupted()
            pixels = np.clip(frame.detach().to(device="cpu", dtype=torch.float32).numpy() * 255, 0, 255).round().astype(np.uint8)
            if pixels.shape[-1] == 1:
                pixels = pixels[..., 0]
            resized = Image.fromarray(pixels).resize((new_width, new_height), getattr(Image.Resampling, method.upper()))
            array = np.array(resized, dtype=np.float32) / 255.0
            if array.ndim == 2:
                array = array[..., None]
            frames.append(torch.from_numpy(array))
        return torch.stack(frames), new_width, new_height


def redact_metadata(value):
    """Redact credentials from copies of prompt/workflow image metadata."""
    result = deepcopy(value)

    def visit(item):
        if isinstance(item, dict):
            if "api_key" in item:
                item["api_key"] = ""
            kind = item.get("type")
            index = {"SeedreamAPIGenerate": 2, "SeedreamAPIGenerateV2": 1}.get(kind)
            values = item.get("widgets_values")
            if index is not None and isinstance(values, list) and len(values) > index:
                values[index] = ""
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(result)
    return result


class SeedreamSaveImage(SaveImage):
    CATEGORY = "Seedream API"
    DESCRIPTION = "预览并保存生成图片，并从 PNG 的工作流信息中移除 API 密钥。"

    def save_images(self, images, filename_prefix="Seedream/Seedream", prompt=None, extra_pnginfo=None):
        return super().save_images(images, filename_prefix, redact_metadata(prompt), redact_metadata(extra_pnginfo))


NODE_CLASS_MAPPINGS = {
    "SeedreamAPIGenerate": SeedreamAPIGenerate,
    "SeedreamAPIGenerateV2": SeedreamAPIGenerateV2,
    "SeedreamPromptText": SeedreamPromptText,
    "SeedreamLongEdge": SeedreamLongEdge,
    "SeedreamResizeLongEdge": SeedreamResizeLongEdge,
    "SeedreamSaveImage": SeedreamSaveImage,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedreamAPIGenerate": "Seedream API · 简洁版 · ¥0.1/次",
    "SeedreamAPIGenerateV2": "Seedream · 图像生成 · ¥0.1/次",
    "SeedreamPromptText": "文本 · 提示词",
    "SeedreamLongEdge": "整数 · 最大边长",
    "SeedreamResizeLongEdge": "按最大边长缩放",
    "SeedreamSaveImage": "保存图像 · Seedream",
}
