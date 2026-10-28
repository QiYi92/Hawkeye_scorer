"""Optional ROI parsers and bounded optical-flow propagation. No keypoints."""
from pathlib import Path

import cv2
import numpy as np


def parsing_device(device):
    import torch
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    return device


class TorchScriptParser:
    """Contract: RGB ImageNet-normalized [1,3,S,S] -> logits [1,3,H,W]."""
    def __init__(self, weights, device="cpu", size=256):
        import torch
        device = parsing_device(device)
        self.torch, self.device, self.size = torch, device, size
        self.model = torch.jit.load(str(weights), map_location=device).eval()

    def parse(self, crop):
        t = self.torch
        rgb = cv2.cvtColor(cv2.resize(crop, (self.size, self.size)), cv2.COLOR_BGR2RGB)
        array = (rgb.astype(np.float32) / 255 - [.485, .456, .406]) / [.229, .224, .225]
        tensor = t.from_numpy(array.astype(np.float32).transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with t.inference_mode():
            logits = self.model(tensor)
        if not isinstance(logits, t.Tensor) or logits.ndim != 4 or logits.shape[:2] != (1, 3):
            raise ValueError("腿脚 TorchScript 模型必须返回 [1,3,H,W] logits：background / legs / feet")
        labels = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
        return cv2.resize(labels, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)


class SCHPParser:
    """Adapter for the official external SCHP repo and LIP checkpoint.

    Uses unmodified upstream affine helpers and the model architecture.
    LIP pants/socks belong to visible legs here; shoes belong to feet.
    """
    def __init__(self, repo, weights, device="cpu"):
        import torch
        repo = Path(repo).resolve()
        if not (repo / "networks/__init__.py").is_file():
            raise ValueError(f"无效 SCHP 仓库：{repo}")
        from .schp_compat import load_schp_modules
        network, transforms = load_schp_modules(repo)
        device = parsing_device(device)
        self.affine, self.restore = transforms.get_affine_transform, transforms.transform_logits
        self.torch, self.device = torch, device
        self.model = network.resnet101(num_classes=20, pretrained=None)
        checkpoint = torch.load(str(weights), map_location="cpu", weights_only=True)
        state = checkpoint.get("state_dict", checkpoint)
        self.model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()})
        self.model.to(device).eval()

    def parse(self, crop):
        t = self.torch
        h, w = crop.shape[:2]
        center = np.array([(w - 1) / 2, (h - 1) / 2], np.float32)
        scale = np.full(2, max(w - 1, h - 1, 1), np.float32)
        size = [473, 473]
        warped = cv2.warpAffine(crop, self.affine(center, scale, 0, size), tuple(size))
        # Upstream SCHP preprocessing intentionally keeps OpenCV BGR channel order.
        arr = (warped.astype(np.float32) / 255 - [.406, .456, .485]) / [.225, .224, .229]
        tensor = t.from_numpy(arr.astype(np.float32).transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with t.inference_mode():
            output = self.model(tensor)[0][-1]
            logits = t.nn.functional.interpolate(output, size=size, mode="bilinear", align_corners=True)[0]
        original = self.restore(logits.permute(1, 2, 0).cpu().numpy(), center, scale, w, h, input_size=size)
        lip = original.argmax(2)
        labels = np.zeros((h, w), np.uint8)
        labels[np.isin(lip, [8, 9, 16, 17])] = 1
        labels[np.isin(lip, [18, 19])] = 2
        return labels


def propagate_labels(previous_gray, current_gray, labels):
    """Backward flow samples previous labels at each current pixel; no label blending."""
    h, w = current_gray.shape
    factor = min(1., 640 / max(h, w))
    small_size = (max(1, round(w * factor)), max(1, round(h * factor)))
    prev = cv2.resize(previous_gray, small_size)
    curr = cv2.resize(current_gray, small_size)
    flow = cv2.calcOpticalFlowFarneback(curr, prev, None, .5, 3, 15, 3, 5, 1.2, 0)
    flow = cv2.resize(flow, (w, h))
    flow[:, :, 0] *= w / small_size[0]
    flow[:, :, 1] *= h / small_size[1]
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    return cv2.remap(labels, xx + flow[:, :, 0], yy + flow[:, :, 1],
                     cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)


class PartMasks:
    def __init__(self, parser=None, interval=3):
        self.parser, self.interval = parser, interval
        self.reset()

    def reset(self):
        self.track_id = self.gray = self.labels = None
        self.last_frame = self.last_inference = -1

    def update(self, frame, instance, frame_id, force_inference=False):
        if self.parser is None:
            return None, "disabled", None
        if instance is None:
            self.reset()
            return None, "unavailable", None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fresh = instance.track_id != self.track_id or self.last_frame != frame_id - 1
        if force_inference or fresh or frame_id - self.last_inference >= self.interval:
            x, y, w, h = cv2.boundingRect(instance.mask.astype(np.uint8))
            # Context padding helps parsing around shoes and harness edges.
            pad = max(2, round(max(w, h) * .08))
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
            labels = np.zeros(gray.shape, np.uint8)
            labels[y1:y2, x1:x2] = self.parser.parse(frame[y1:y2, x1:x2])
            self.last_inference = frame_id
            source = "inferred"
        else:
            labels = propagate_labels(self.gray, gray, self.labels)
            source = "propagated"
        labels[~instance.mask] = 0
        self.labels, self.gray = labels, gray
        self.track_id, self.last_frame = instance.track_id, frame_id
        return labels, source, frame_id - self.last_inference
