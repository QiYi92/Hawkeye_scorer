"""SCHP inference without compiling its legacy CUDA InPlaceABN extension.

The upstream inference kernel uses gamma=abs(weight)+eps, not plain BN weight.
Keep that convention and checkpoint keys; training is deliberately unsupported.
"""
import ast
import importlib.util
from types import ModuleType

import torch
from torch import nn
from torch.nn import functional as F


class InPlaceABNSync(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=.1, affine=True,
                 activation="leaky_relu", slope=.01):
        super().__init__()
        self.eps, self.activation, self.slope = eps, activation, slope
        if activation not in ("none", "relu", "leaky_relu", "elu"):
            raise ValueError(f"Unsupported SCHP activation: {activation}")
        self.weight = nn.Parameter(torch.ones(num_features)) if affine else None
        self.bias = nn.Parameter(torch.zeros(num_features)) if affine else None
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        if self.training:
            raise RuntimeError("SCHP compatibility normalization is inference-only; call eval()")
        gamma = self.weight.abs() + self.eps if self.weight is not None else None
        x = F.batch_norm(x, self.running_mean, self.running_var, gamma, self.bias,
                         training=False, eps=self.eps)
        if self.activation == "leaky_relu":
            return F.leaky_relu(x, negative_slope=self.slope)
        if self.activation == "relu":
            return F.relu(x)
        if self.activation == "elu":
            return F.elu(x)
        return x


def load_schp_modules(repo):
    """Load upstream files privately, replacing only the legacy extension import.

    No upstream file edits, global sys.path additions, or generic 'utils/modules'
    entries in sys.modules; avoid collisions with other models in this process.
    """
    path = repo / "networks/AugmentCE2P.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    replaced = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "modules":
            if [(alias.name, alias.asname) for alias in node.names] != [("InPlaceABNSync", None)]:
                raise ValueError("SCHP 源码的 modules 接口已变化，不能安全应用兼容层")
            node.module = "demo13.schp_compat"
            replaced += 1
    if replaced != 1:
        raise ValueError("SCHP 源码版本不匹配：未找到唯一的 InPlaceABNSync 导入")
    network = ModuleType("_demo13_schp_network")
    network.__file__ = str(path)
    exec(compile(tree, str(path), "exec"), network.__dict__)
    transforms_path = repo / "utils/transforms.py"
    spec = importlib.util.spec_from_file_location("_demo13_schp_transforms", transforms_path)
    transforms = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transforms)
    return network, transforms
