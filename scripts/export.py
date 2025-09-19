import argparse
from pathlib import Path
from typing import Any

import torch

from src.models import build_mobileface
from src.utils.config import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the MobileFaceNet model to ONNX.")
    parser.add_argument("--config", type=str, default="configs/mobile.yaml", help="Path to the model configuration.")
    parser.add_argument("--weights", type=str, required=True, help="Path to the trained model weights (state dict).")
    parser.add_argument("--output", type=str, default="artifacts/exports/mobileface.onnx", help="Output ONNX file path.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to load the model on during export.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument("--image-size", type=int, default=112, help="Input resolution for dummy tracing tensor.")
    parser.add_argument("--quantize", action="store_true", help="Run post-training dynamic quantization on the exported graph.")
    return parser.parse_args()


def resolve_path(base_dir: Path, candidate: str | Path) -> Path:
    candidate_path = Path(candidate)
    if candidate_path.is_absolute():
        return candidate_path
    return (base_dir / candidate_path).resolve()


def build_model(config: dict[str, Any]) -> torch.nn.Module:
    model_cfg = config.get("model", {})
    return build_mobileface(
        embedding_dim=int(model_cfg.get("embedding_dim", 128)),
        width_multiplier=float(model_cfg.get("width_multiplier", 1.0)),
        dropout=float(model_cfg.get("backbone_dropout", 0.0)),
    )


def export_onnx(model: torch.nn.Module, dummy_input: torch.Tensor, output_path: Path, opset: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["images"],
        output_names=["embeddings"],
        opset_version=opset,
        dynamic_axes={"images": {0: "batch"}, "embeddings": {0: "batch"}},
    )


def quantize_model(onnx_path: Path) -> Path:
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as exc:
        raise RuntimeError("onnxruntime is required for quantization. Install it via requirements.txt.") from exc

    quant_path = onnx_path.with_suffix(".int8.onnx")
    quantize_dynamic(model_input=str(onnx_path), model_output=str(quant_path), weight_type=QuantType.QInt8)
    return quant_path


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    weights_path = Path(args.weights).resolve()
    device = torch.device(args.device)

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    config = load_yaml_config(config_path)
    config_dir = config_path.parent
    model = build_model(config)
    state_dict = torch.load(weights_path, map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: missing keys in state dict: {missing}")
    if unexpected:
        print(f"Warning: unexpected keys in state dict: {unexpected}")
    model.to(device)
    model.eval()

    dummy_input = torch.randn(1, 3, args.image_size, args.image_size, device=device)
    output_path = resolve_path(config_dir, args.output)
    export_onnx(model, dummy_input, output_path, args.opset)
    print(f"Exported ONNX model to {output_path}")

    if args.quantize:
        quant_path = quantize_model(output_path)
        print(f"Quantized model written to {quant_path}")


if __name__ == "__main__":
    main()
