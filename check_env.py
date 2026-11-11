#!/usr/bin/env python
"""Standalone XAIP environment check — run to verify an environment is usable.

    python check_env.py           # CPU check
    python check_env.py --gpu     # additionally require a usable CUDAExecutionProvider

No project imports, no config, no weights needed. Checks:
  1. Python version (environment.yml requires 3.12).
  2. Every pip-pinned dependency in environment.yml imports.
  3. Installed versions match the exact pins (WARN on drift; a GPU pip install
     legitimately replaces torch/torchvision/onnxruntime — local build tags
     like +cpu/+cu118 are normalized).
  4. ONNX Runtime execution providers (CPU always; CUDA optional / required with --gpu).
  5. Executes a tiny synthetic ONNX graph (proves onnx + onnxruntime actually run).

Exit code 0 only when the environment is usable (CI-friendly).
"""

import argparse
import importlib
import re
import sys
from importlib.metadata import version as dist_version
from pathlib import Path

EXPECTED_PY = (3, 12)

# (import name, distribution name) — order mirrors environment.yml
PINNED = [
    ("numpy", "numpy"),
    ("cv2", "opencv-python"),
    ("onnx", "onnx"),
    ("onnxruntime", "onnxruntime"),
    ("yaml", "PyYAML"),
    ("PIL", "Pillow"),
    ("sklearn", "scikit-learn"),
    ("matplotlib", "matplotlib"),
    ("pandas", "pandas"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("lap", "lap"),
    ("cython_bbox", "cython_bbox"),
]

# Distributions that the guidelined GPU pip install legitimately replaces
# (GUIDELINE § 1) — all other drift falls back to the generic re-create hint.
GPU_PIP_OVERRIDES = {"torch", "torchvision", "onnxruntime"}

_ENV_FILE = Path(__file__).resolve().parent / "environment.yml"
_WIDTH = max(len(name) for name, _ in PINNED) + 2
_IS_TTY = sys.stdout.isatty()
_PIN_RE = re.compile(r"^\s*-\s*([A-Za-z0-9_.\-]+)==([0-9][A-Za-z0-9._+\-]*)\s*$")


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _IS_TTY else text


def _ok(text: str = "OK"):
    return _c("32", f"[{text}]")


def _warn(text: str = "WARN"):
    return _c("33", f"[{text}]")


def _fail(text: str = "FAIL"):
    return _c("31", f"[{text}]")


def _line(label: str, status: str, detail: str = ""):
    row = f"  {label:<{_WIDTH}}{status}"
    if detail:
        row += f"  {detail}"
    print(row.rstrip())


def check_python() -> bool:
    got = ".".join(str(part) for part in sys.version_info[:3])
    expected = ".".join(str(part) for part in EXPECTED_PY)
    ok = sys.version_info[:2] == EXPECTED_PY
    _line("python", _ok() if ok else _fail(), f"{got} (expected {expected}, environment.yml)")
    return ok


def read_pins() -> dict:
    if not _ENV_FILE.exists():
        return {}
    pins = {}
    for line in _ENV_FILE.read_text().splitlines():
        match = _PIN_RE.match(line)
        if match:
            pins[match.group(1).lower()] = match.group(2)
    return pins


def _pinned_eq(installed: str, target: str) -> bool:
    # Ignore local build tags: +cpu builds, +cu118 GPU wheels, etc.
    if "+" in installed and "+" not in target:
        installed = installed.split("+", 1)[0]
    return installed == target


def check_imports(pins: dict) -> dict:
    modules = {}
    for name, dist in PINNED:
        try:
            modules[name] = importlib.import_module(name)
        except Exception as exc:
            _line(name, _fail(), f"{dist}: {type(exc).__name__}: {exc}")
            modules[name] = None
            continue
        try:
            installed = dist_version(dist)
        except Exception:
            installed = getattr(modules[name], "__version__", "?")
        target = pins.get(dist.lower())
        if target and not _pinned_eq(installed, target):
            if dist in GPU_PIP_OVERRIDES:
                hint = "expected if you ran the GPU pip install (GUIDELINE § 1)"
            else:
                hint = f"environment drifted from the pins — `pip install {dist}=={target}` to realign"
            _line(name, _warn(), f"{installed} != pinned {target} — {hint}")
        else:
            _line(name, _ok(), f"{installed}" + (f" (== {target})" if target else ""))
    return modules


def check_providers(ort, require_gpu: bool):
    providers = [p.replace("ExecutionProvider", "") for p in ort.get_available_providers()]
    cpu_ok = "CPU" in providers
    cuda_ok = "CUDA" in providers
    _line("providers", _ok() if cpu_ok else _fail(), ", ".join(providers))
    if not cuda_ok:
        if require_gpu:
            _line("cuda", _fail(), "GPU runs would silently stay on CPU — install onnxruntime-gpu per GUIDELINE § 1 (CUDA 11.8)")
        else:
            _line("cuda", _warn(), "CPU only — GPU runs would silently fall back to CPU despite target.device: \"gpu\"")
    return cpu_ok and (cuda_ok if require_gpu else True), cuda_ok


def run_smoke(modules: dict, has_cuda: bool) -> bool:
    np, onnx, ort = modules["numpy"], modules["onnx"], modules["onnxruntime"]
    if not all((np, onnx, ort)):
        _line("graph exec", _fail(), "skipped — numpy/onnx/onnxruntime did not import")
        return False
    from onnx import TensorProto, helper, numpy_helper

    try:
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 8])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4, 8])
        graph = helper.make_graph(
            [helper.make_node("Add", ["x", "c"], ["y"])], "smoke", [x], [y],
            initializer=[numpy_helper.from_array(np.ones([4, 8], np.float32), "c")])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        onnx.checker.check_model(model)
        wanted = ["CUDAExecutionProvider", "CPUExecutionProvider"] if has_cuda else ["CPUExecutionProvider"]
        session = ort.InferenceSession(model.SerializeToString(), providers=wanted)
        used = "+".join(p.replace("ExecutionProvider", "") for p in session.get_providers())
        out = session.run(None, {"x": np.zeros([4, 8], np.float32)})[0]
        assert np.allclose(out, 1.0)
    except Exception as exc:
        _line("graph exec", _fail(), f"{type(exc).__name__}: {exc}")
        return False
    _line("graph exec", _ok(), f"Add(4x8) ran via {used}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check the XAIP runtime environment (imports, pinned versions, ORT providers, smoke run).")
    parser.add_argument("--gpu", action="store_true",
                        help="exit non-zero unless CUDAExecutionProvider is available")
    args = parser.parse_args()

    print(f"XAIP env check — {sys.executable}")
    pins = read_pins()

    failures = not check_python()
    modules = check_imports(pins)
    if any(module is None for module in modules.values()):
        failures = True

    if modules.get("onnxruntime"):
        providers_ok, cuda_ok = check_providers(modules["onnxruntime"], args.gpu)
        providers_ok = providers_ok and run_smoke(modules, cuda_ok)
        failures |= not providers_ok

    print()
    if failures:
        print(_fail("BROKEN") + "  fix: `pip install -r requirements.txt` (venv), `conda env create -f environment.yml` or `./run.sh` — see GUIDELINE § 1")
        return 1
    print(_ok("READY") + "  environment is usable — continue with GUIDELINE § 2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
