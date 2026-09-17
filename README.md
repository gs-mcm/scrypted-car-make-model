# Vehicle Make/Model Classifier for Scrypted (CoreML)

A Scrypted "custom detection" package that identifies vehicle make, model and
generation (e.g. `Toyota Camry (XV50) 2012-2017`) on Apple Silicon Macs via the
`@scrypted/coreml` plugin.

The model is **AutoVision v5.13.0** by bmoldo (https://github.com/bmoldo/carvision):
EfficientNet-V2-S, 384x384 RGB input, 897 classes (896 model-generations across
76 makes + `background`). Reported accuracy: 93.85% top-1 / 97.88% top-5.
Weights are FP16 (42 MB). Inference on an M-series Mac is ~3-10 ms per crop.

## License

The model weights are **PolyForm Noncommercial 1.0.0** (see `LICENSE-AutoVision.txt`).
Free for personal / non-commercial use. Commercial use needs a license from the
author (`LICENSE-COMMERCIAL-AutoVision.md`).

## Layout (what Scrypted expects)

```
models/coreml/config.json          <- Scrypted reads this: model type, input shape, file list, labels
models/coreml/model.mlpackage/     <- the CoreML model (unzipped; Scrypted downloads each file)
class_mapping.json                 <- original AutoVision taxonomy (make/model/generation/years/rarity)
model_manifest.json                <- original AutoVision manifest
```

`config.json` uses Scrypted's `"model": "resnet"` classifier path: Scrypted resizes the
object crop to 384x384, feeds the image to the model, applies softmax to the logits,
and reports up to 3 classes scoring above 0.5. `mean`/`std` are intentionally
**not** set because this mlpackage takes an image input with ImageNet normalization
baked in; Scrypted only passes a raw PIL image when mean/std are absent.

## Installing into Scrypted

Scrypted fetches the model over HTTP from a **Model URL**. Two ways to host it:

### Option A: GitHub repo (the way Scrypted's reference bird-classifier is published)

This repo is already laid out the way Scrypted expects.

1. In Scrypted: **Plugins -> CoreML Object Detection -> Add Model** (the plugin's
   "Create Device" form). Fill in:
   - Model Name: `Vehicle Classifier`
   - Model URL: `https://github.com/gs-mcm/scrypted-car-make-model`
   Scrypted turns that into
   `https://raw.githubusercontent.com/gs-mcm/scrypted-car-make-model/refs/heads/main/models/coreml/config.json`
   (you can also paste that raw URL directly).

### Option B: serve it from the Scrypted Mac itself (no GitHub needed)

1. Copy this folder to the Mac running Scrypted, e.g. `~/vehicle-classifier`.
2. Serve it over HTTP. From that folder:

   ```bash
   python3 -m http.server 8765 --bind 127.0.0.1
   ```

   (Keep it running until the plugin has downloaded the files; afterwards Scrypted
   uses its cached copy in the plugin volume. Use a launchd job if you want it
   permanent so re-installs can re-download.)
3. In Scrypted: **Plugins -> CoreML Object Detection -> Add Model**:
   - Model Name: `Vehicle Classifier`
   - Model URL: `http://127.0.0.1:8765/models/coreml/config.json`
   (A URL ending in `config.json` is used as-is.)

After adding, the model appears under **CoreML Object Detection -> Models** as a
device with a **Settings** page. Its only setting is **Exclude Classes**. Add
`background` there if you never want "not a car" to be reported.

Check the CoreML plugin console: on first load it logs `Downloading ...` for the three
mlpackage files, then loads the model.

## How it is used by Scrypted NVR

Custom detection devices expose the `CustomObjectDetection` interface. Scrypted NVR
(closed source) runs them on the crops of objects found by the primary detector,
so this classifier sees each detected `vehicle` and adds the make/model class. The
exact surfaces where the label shows up (detection labels, AI descriptions,
notification filters) are controlled by the NVR plugin and change between releases;
see https://github.com/koush/scrypted/discussions/1992 for a user report that class
exclusions did not (at that time) feed into notification triggers.

## Verifying the model outside Scrypted

```bash
python3.12 -m venv venv && ./venv/bin/pip install coremltools pillow numpy
./venv/bin/python - <<'PY'
import coremltools as ct, numpy as np, json
from PIL import Image
cfg = json.load(open("models/coreml/config.json"))
m = ct.models.MLModel("models/coreml/model.mlpackage")
im = Image.open("some_car.jpg").convert("RGB").resize((384, 384))
logits = np.asarray(list(m.predict({"image": im}).values())[0][0], dtype=np.float32)
p = np.exp(logits - logits.max()); p /= p.sum()
for i in np.argsort(p)[::-1][:3]:
    print(cfg["labels"][str(i)], round(float(p[i]), 3))
PY
```

Note: coremltools needs Python 3.12 or 3.13 (its native libs are not published for 3.14).

## Label format

Labels are derived from `class_mapping.json` as `Make Model (Generation) start-end`, e.g.
`Ford F-150 (13th Gen) 2014-2020`, `Jeep Wrangler (JK) 2006-2018`. Index 896 is `background`.
Edit `models/coreml/config.json` if you prefer different display names (keep all 897 keys).
