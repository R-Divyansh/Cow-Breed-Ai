import io
import traceback
import base64
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, render_template
from PIL import Image
import os

from model import CowBreedClassifier

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIDENCE_THRESHOLD = 0.60   # 10% — raise once model is fully trained

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────
print("Loading model...")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "cow_web_model.pth")

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
CLASS_NAMES = checkpoint["classes"]
NUM_CLASSES = len(CLASS_NAMES)
print(f"Classes : {CLASS_NAMES}")
print(f"Epoch   : {checkpoint.get('epoch', 'unknown')}")
print(f"Val acc : {checkpoint.get('val_acc', 'unknown')}")

model = CowBreedClassifier(
    num_classes=NUM_CLASSES,
    pretrained=False,
    unet_base_ch=32,
    feat_mid_channels=32,
    feat_n_blocks=8
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print("Model ready.\n")

# ─────────────────────────────────────────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────────────────────────────────────────
# IMPORTANT: The gradient hook is only registered INSIDE generate_gradcam(),
# not at startup. This is because torch.no_grad() (used during normal inference)
# strips gradient tracking from all tensors — a hook on a no-grad tensor raises:
#   RuntimeError: cannot register a hook on a tensor that doesn't require gradient
#
# Solution: during Grad-CAM we run a SEPARATE forward pass with grad ENABLED
# (no torch.no_grad() wrapper), register the hooks fresh each time, then
# remove them immediately after so they don't interfere with inference.

def generate_gradcam(tensor: torch.Tensor, class_idx: int) -> str:
    """
    Runs a grad-enabled forward+backward pass to produce a Grad-CAM heatmap.
    Hooks are registered and removed within this function only.
    Returns base64 PNG string, or '' if anything fails.
    """
    try:
        import numpy as np

        activations = {}
        gradients   = {}

        # Register hooks freshly for this call only
        def fwd_hook(module, input, output):
            activations["feat"] = output   # keep as-is (grad-capable tensor)

        def bwd_hook(module, input, output):
            # output[0] is the grad of the norm layer's input
            gradients["feat"] = output[0].detach()

        # attach
        target_layer = model.backbone.layers[-1].blocks[-1].norm1

        fwd_handle = target_layer.register_forward_hook(fwd_hook)
        bwd_handle = target_layer.register_full_backward_hook(bwd_hook)

        try:
            model.zero_grad()
            # grad-enabled pass — NO torch.no_grad() here
            logits, _ = model(tensor)
            score     = logits[0, class_idx]
            score.backward()
        finally:
            # always remove hooks so they don't pollute future inference calls
            fwd_handle.remove()
            bwd_handle.remove()

        act  = activations.get("feat")
        grad = gradients.get("feat")

        if act is None or grad is None:
            print("Grad-CAM: hooks returned None — skipping")
            return ""

        # act  shape: (1, num_tokens, C)  e.g. (1, 49, 768)
        # grad shape: (1, num_tokens, C)
        weights = grad.mean(dim=1).squeeze(0)           # (C,)
        cam     = (act.detach()[0] * weights).sum(dim=-1)  # (num_tokens,)
        cam     = F.relu(cam)

        num_tokens = cam.shape[0]

        # Try safe square reshape
        side = int(num_tokens ** 0.5)

        if side * side != num_tokens:
            print(f"Grad-CAM: non-square tokens ({num_tokens}), using fallback 7x7")
            side = 7
            cam = cam[:side * side]

        cam = cam.reshape(side, side)
        cam  = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        cam_np = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(224, 224), mode='bilinear', align_corners=False
        ).squeeze().detach().numpy()

        # Jet-like colormap (no matplotlib needed)
        r = np.clip(1.5 - np.abs(cam_np * 4 - 3), 0, 1)
        g = np.clip(1.5 - np.abs(cam_np * 4 - 2), 0, 1)
        b = np.clip(1.5 - np.abs(cam_np * 4 - 1), 0, 1)
        heat = Image.fromarray(np.stack([
            (r*255).astype(np.uint8),
            (g*255).astype(np.uint8),
            (b*255).astype(np.uint8)
        ], axis=-1), "RGB")

        # Reconstruct original image from normalised tensor
        orig_np = ((tensor.squeeze(0) * 0.5 + 0.5) * 255)
        orig_np = orig_np.permute(1, 2, 0).byte().detach().numpy()
        orig    = Image.fromarray(orig_np, "RGB")

        blended = Image.blend(orig, heat, alpha=0.45)
        buf = io.BytesIO()
        blended.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    except Exception as e:
        print(f"Grad-CAM failed (non-fatal): {e}")
        traceback.print_exc()
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# NUMPY-FREE PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(image: Image.Image) -> torch.Tensor:
    """PIL → (1,3,224,224) normalised tensor. No NumPy."""
    image  = image.resize((224, 224), Image.BILINEAR)
    raw    = bytearray(image.tobytes())
    tensor = torch.frombuffer(raw, dtype=torch.uint8).clone().reshape(224, 224, 3)
    tensor = tensor.permute(2, 0, 1).float() / 255.0
    tensor = (tensor - 0.5) / 0.5
    return tensor.unsqueeze(0)


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", classes=CLASS_NAMES)


@app.route("/predict", methods=["POST"])
def predict():
    print("\n=== /predict ===")
    files = request.files.getlist("images")
    print(f"Files: {[f.filename for f in files]}")

    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "No images uploaded"}), 400

    all_probs    = []
    first_tensor = None

    for i, f in enumerate(files[:5]):
        print(f"\n-- File {i+1}: {f.filename} --")
        try:
            raw   = f.read()
            image = Image.open(io.BytesIO(raw)).convert("RGB")
            print(f"  PIL size : {image.size}")

            tensor = preprocess(image)
            print(f"  Tensor   : {tensor.shape}  min={tensor.min():.2f} max={tensor.max():.2f}")

            if first_tensor is None:
                first_tensor = tensor.clone()

            # ── Normal inference — no_grad is safe here because we do NOT
            #    register any gradient hooks during this block ──────────────
            with torch.no_grad():
                logits, mask = model(tensor)
                probs = F.softmax(logits, dim=1)[0]

            score_str = " | ".join(
                f"{CLASS_NAMES[j]}={probs[j].item()*100:.1f}%"
                for j in range(NUM_CLASSES)
            )
            print(f"  Scores   : {score_str}")
            print(f"  Best     : {probs.max().item()*100:.1f}% → {CLASS_NAMES[probs.argmax().item()]}")
            print(f"  Mask avg : {mask.mean().item():.4f}")

            all_probs.append(probs)

        except Exception as e:
            print(f"  EXCEPTION: {e}")
            traceback.print_exc()
            return jsonify({"error": f"Failed on {f.filename}: {str(e)}"}), 500

    if not all_probs:
        return jsonify({"error": "No images could be processed"}), 400

    avg_probs = torch.stack(all_probs).mean(dim=0)
    max_conf  = avg_probs.max().item()
    print(f"\nFinal max conf : {max_conf*100:.1f}%  (threshold={CONFIDENCE_THRESHOLD*100:.0f}%)")

    # ── Threshold guard ───────────────────────────────────────────────────────
    if max_conf < CONFIDENCE_THRESHOLD:
        ranked = sorted(
            [(CLASS_NAMES[i], round(avg_probs[i].item()*100, 1)) for i in range(NUM_CLASSES)],
            key=lambda x: -x[1]
        )
        return jsonify({
            "valid":          False,
            "reason": (
                f"Confidence is only {max_conf*100:.1f}% — below the "
                f"{CONFIDENCE_THRESHOLD*100:.0f}% threshold. "
                f"Top scores: {', '.join(f'{b}: {c}%' for b,c in ranked[:3])}. "
                f"Try a clearer, well-lit photo with the cow centred in frame."
            ),
            "max_confidence": round(max_conf * 100, 1),
            "all_scores":     dict(ranked)
        })

    # ── Top-3 predictions ─────────────────────────────────────────────────────
    top3_vals, top3_idx = torch.topk(avg_probs, k=min(3, NUM_CLASSES))
    predictions = [
        {"breed": CLASS_NAMES[idx.item()], "confidence": round(val.item()*100, 2), "rank": i+1}
        for i, (val, idx) in enumerate(zip(top3_vals, top3_idx))
    ]
    print(f"Top prediction : {predictions[0]}")

    # ── Grad-CAM — separate grad-enabled pass, safe after no_grad inference ──
    gradcam_b64 = ""
    if first_tensor is not None:
        print("Generating Grad-CAM...")
        gradcam_b64 = generate_gradcam(first_tensor, top3_idx[0].item())
        print(f"Grad-CAM: {'generated OK' if gradcam_b64 else 'skipped'}")

    return jsonify({
        "valid":       True,
        "predictions": predictions,
        "gradcam":     gradcam_b64,
        "num_images":  len(all_probs),
        "top_breed":   CLASS_NAMES[top3_idx[0].item()],
        "top_conf":    round(max_conf * 100, 2)
    })


@app.route("/debug", methods=["POST"])
def debug():
    """No-threshold debug — returns raw scores for every breed."""
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No image"}), 400
    try:
        image  = Image.open(io.BytesIO(files[0].read())).convert("RGB")
        tensor = preprocess(image)
        with torch.no_grad():
            logits, mask = model(tensor)
            probs = F.softmax(logits, dim=1)[0]
        ranked = sorted(
            [(CLASS_NAMES[i], round(probs[i].item()*100, 2)) for i in range(NUM_CLASSES)],
            key=lambda x: -x[1]
        )
        return jsonify({
            "all_scores":      dict(ranked),
            "max_confidence":  round(probs.max().item()*100, 2),
            "predicted_class": CLASS_NAMES[probs.argmax().item()],
            "threshold_%":     CONFIDENCE_THRESHOLD * 100,
            "would_pass":      bool(probs.max().item() >= CONFIDENCE_THRESHOLD),
            "mask_mean":       round(mask.mean().item(), 4)
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)