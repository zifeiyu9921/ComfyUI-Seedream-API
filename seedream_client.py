"""Adapted from the user's seedream_client.py; API fields are unchanged."""

from contextlib import ExitStack
from fractions import Fraction
from pathlib import Path
from urllib.parse import urljoin, urlsplit
import os
import re
import time

import requests
from PIL import Image


ASPECT_RATIOS = ("1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9")
MAX_IMAGE_BYTES = 100 * 1024 * 1024


def require_api_key(value):
    """Require an explicit key; only the remote API decides whether it is valid."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("密钥不能为空，请在生成节点中输入有效的云端 API 密钥。")
    return value.strip()


def closest_image_ratio(image):
    width, height = image.size
    if image.getexif().get(274, 1) in {5, 6, 7, 8}:
        width, height = height, width
    if width <= 0 or height <= 0:
        raise ValueError("图片尺寸无效，无法识别比例。")

    def mismatch(option):
        target_width, target_height = map(int, option.split(":"))
        actual, target = width * target_height, height * target_width
        return Fraction(max(actual, target), min(actual, target))

    return min(ASPECT_RATIOS, key=mismatch)


class SeedreamClient:
    def __init__(self, base_url=None, api_key=None):
        self.api_key = require_api_key(api_key)
        self.base_url = (base_url or os.getenv("SEEDREAM_BASE_URL") or "https://a6.zifeiyuai.top:2345").rstrip("/")
        base = urlsplit(self.base_url)
        if base.scheme not in {"http", "https"} or not base.netloc or base.username or base.password or base.query or base.fragment:
            raise ValueError("Seedream 接口地址格式无效。")
        self.session = requests.Session()
        self.session.headers["X-API-Key"] = self.api_key

    def close(self):
        self.session.close()

    def _message(self, payload, fallback):
        message = str(payload.get("message") or fallback)
        return message.replace(self.api_key, "***") if self.api_key else message

    @staticmethod
    def _check_response(response):
        if 300 <= response.status_code < 400:
            raise RuntimeError("模型接口返回了不支持的跳转。")
        if response.status_code in (401, 403):
            raise RuntimeError("云端密钥验证失败，请在节点中输入有效的 API 密钥。")
        response.raise_for_status()

    def _json(self, response):
        self._check_response(response)
        try:
            result = response.json()
        except ValueError:
            raise RuntimeError("模型接口没有返回有效 JSON，请检查接口地址和服务状态。") from None
        if not isinstance(result, dict) or "code" not in result:
            raise RuntimeError("模型接口返回格式不正确：缺少状态码 code。")
        return result

    def generate(self, prompt, images=(), ratio="auto", output_dir="seedream_results", timeout=3000,
                 on_progress=None, poll_interval=2, check_cancel=None):
        cancel = check_cancel or (lambda: None)
        if isinstance(images, (str, Path)):
            images = [images]
        images = list(images)
        ratio = str(ratio or "auto").strip().lower()
        if ratio not in ("auto", *ASPECT_RATIOS):
            raise ValueError("当前模型不支持此图片比例，请重新选择。")
        if ratio == "auto" and images:
            with Image.open(images[0]) as image:
                ratio = closest_image_ratio(image)
        cancel()
        with ExitStack() as stack:
            files = [("file", (Path(p).name, stack.enter_context(open(p, "rb")))) for p in images]
            # Never retry this POST automatically: it could create duplicate paid tasks.
            with self.session.post(self.base_url + "/add_task", data={"prompt": prompt, "ratio": ratio},
                                   files=files, timeout=(15, 120), allow_redirects=False) as response:
                result = self._json(response)
        if result["code"] != 0:
            raise RuntimeError(self._message(result, "任务提交失败。"))
        taskid = result.get("taskid")
        if not isinstance(taskid, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,160}", taskid):
            raise RuntimeError("模型返回的任务编号无效。")
        if on_progress:
            on_progress({"progress": f"任务已提交：{taskid}"})
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            cancel()
            remaining = max(0.1, deadline - time.monotonic())
            try:
                with self.session.get(self.base_url + "/try_get", params={"taskid": taskid},
                                      timeout=(min(10, remaining), min(30, remaining)), allow_redirects=False) as response:
                    task = self._json(response)
            except (requests.Timeout, requests.ConnectionError):
                # Retry only the read-only status query; keep the existing task id.
                task = {"code": 1, "progress": f"查询暂时失败，继续等待任务：{taskid}"}
            if on_progress:
                on_progress(task)
            if task["code"] == -1:
                raise RuntimeError(self._message(task, "模型生成失败。"))
            if task["code"] == 0:
                break
            wait_until = min(deadline, time.monotonic() + poll_interval)
            while time.monotonic() < wait_until:
                cancel()
                time.sleep(min(0.2, max(0, wait_until - time.monotonic())))
        else:
            raise TimeoutError(f"等待超时，服务端任务可能仍在运行，任务编号：{taskid}")
        results = task.get("results")
        if not isinstance(results, list) or not results:
            raise RuntimeError("模型未返回生成图片。")
        folder = Path(output_dir) / taskid
        folder.mkdir(parents=True, exist_ok=True)
        paths = []
        for index, item in enumerate(results, 1):
            cancel()
            if not isinstance(item, dict) or not isinstance(item.get("url"), str) or not item["url"]:
                raise RuntimeError("模型返回的图片下载信息无效。")
            url = urljoin(self.base_url + "/", item["url"])
            base, download = urlsplit(self.base_url), urlsplit(url)
            if (download.scheme, download.netloc) != (base.scheme, base.netloc):
                raise RuntimeError("模型返回的下载地址无效：仅允许下载同一接口站点的图片。")
            suffix = Path(str(item.get("name", ""))).suffix.lower()
            if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
                raise RuntimeError("模型返回了不支持的图片格式。")
            target = folder / f"{index}{suffix}"
            partial = target.with_suffix(target.suffix + ".part")
            try:
                with self.session.get(url, stream=True, timeout=(15, 120), allow_redirects=False) as response:
                    self._check_response(response)
                    with partial.open("wb") as output:
                        size = 0
                        for block in response.iter_content(1024 * 1024):
                            cancel()
                            size += len(block)
                            if size > MAX_IMAGE_BYTES:
                                raise RuntimeError("模型返回的图片文件超过 100 MB。")
                            output.write(block)
                with Image.open(partial) as image:
                    image.verify()
                partial.replace(target)
            finally:
                partial.unlink(missing_ok=True)
            paths.append(str(target.resolve()))
        return paths
