import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image
import torch
from torch import nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

from preprocess import CLASSES, prepare

ROOT = Path(__file__).resolve().parent
SEED = 20261117


def read_split(root, rows, split):
    samples, labels = [], []
    for row in rows:
        if row["split"] != split:
            continue
        with Image.open(root / row["relative_path"]) as image:
            samples.append(prepare(image))
        labels.append(CLASSES.index(row["class_label"]))
    return torch.from_numpy(np.stack(samples)), torch.tensor(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest = args.dataset / "manifest.csv"
    with manifest.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    groups = {split: {row["origin_group"] for row in rows if row["split"] == split}
              for split in ("train", "val", "test")}
    assert sum(map(len, groups.values())) == len(set.union(*groups.values()))
    assert all(row["variant"] == "original" for row in rows if row["split"] != "train")
    for row in rows:
        path = args.dataset / row["relative_path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Archivo alterado: {path}")

    train_x, train_y = read_split(args.dataset, rows, "train")
    val_x, val_y = read_split(args.dataset, rows, "val")

    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASSES))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    best_key = (-1.0, float("-inf"))
    model_path = ROOT / "models" / "abejas.onnx"
    history_path = ROOT / "historial.jsonl"
    history_path.write_text("", encoding="utf-8")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for indices in torch.randperm(len(train_y)).split(96):
            optimizer.zero_grad()
            logits = model(train_x[indices].to(device))
            loss = loss_fn(logits, train_y[indices].to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(indices)
        model.eval()
        with torch.inference_mode():
            predictions = torch.cat([model(batch.to(device)).cpu() for batch in val_x.split(96)])
        score = (predictions.argmax(1) == val_y).float().mean().item()
        val_loss = loss_fn(predictions, val_y).item()
        key = (score, -val_loss)
        record = {"epoch": epoch + 1, "train_loss": total_loss / len(train_y),
                  "val_accuracy": score, "val_loss": val_loss}
        history.append(record)
        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print(record, flush=True)
        if key > best_key:
            best_key = key
            best_epoch = epoch + 1
            best_state = {name: value.detach().cpu().clone()
                          for name, value in model.state_dict().items()}

    model.cpu().load_state_dict(best_state)
    model.eval()
    torch.onnx.export(model, val_x[:1], str(model_path), input_names=["image"],
                      output_names=["logits"], opset_version=17, dynamo=False,
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}})
    onnx.checker.check_model(str(model_path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        native = model(val_x).numpy()
    exported = session.run(None, {"image": val_x.numpy()})[0]
    np.testing.assert_allclose(native, exported, rtol=1e-4, atol=1e-4)
    metadata = {"model_version": "1.1.0", "selection": "max val_accuracy, min val_loss, earliest epoch if equal", "seed": SEED, "best_epoch": best_epoch,
                "train_device": device, "counts": {s: sum(r["split"] == s for r in rows) for s in groups},
                "groups": {s: len(g) for s, g in groups.items()}, "history": history,
                "max_logit_error_validation": float(np.abs(native - exported).max()),
                "dataset_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                "torch_version": torch.__version__}
    (ROOT / "models" / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

