# ICL-Noise-UNet: In-Context Learning with Noise Modulation for Ultrasound Segmentation

This project implements **ICL-Noise-UNet**, a U-Net-based architecture enhanced with in-context learning and noise modulation for ultrasound image segmentation. The model leverages contextual information from similar images and incorporates noise maps (residual and local variance) to improve segmentation performance on noisy ultrasound data.

---

## Features

- **ICL-Noise-UNet Architecture**: Custom U-Net with context integration and noise modulation.
- **In-Context Learning (ICL)**: Uses context images and masks to guide target image segmentation.
- **Noise Modulation**: Integrates residual noise and local variance maps into feature learning.
- **Training + Testing Pipeline**: End-to-end workflow in a single script.
- **Multi-Dataset Support**: BUSI, BUSBRA, CAMUS, JNU, Radboud-HC.

---

## Installation

```bash
git clone https://github.com/your-repo/ICL-Noise_UNet.git
cd ICL-Noise_UNet
pip install -r requirements.txt
```

**Requirements**
- Python 3.8+
- PyTorch ≥ 1.10
- PyTorch Lightning
- NumPy, OpenCV, Matplotlib

---

## Usage

### Training and Testing

Training and testing are both handled inside **train_iclnoise.py**.

```bash
python train_iclnoise.py
```

What this script does:
- Loads and augments ultrasound datasets
- Builds ICL-Noise-UNet
- Trains using PyTorch Lightning
- Calculates Dice, IoU, accuracy, precision, recall, and specificity
- Saves checkpoints and visual results

TensorBoard:
```bash
tensorboard --logdir lightning_logs/
```

---


Outputs:
- Dice and IoU scores
- CSV summaries
- Per-model performance reports

---

## Model Architectures

### ICL-Noise-UNet
**File:** `models/NoiseContext.py`

Core components:
- Context-aware encoder-decoder
- Noise modulation blocks
- Residual + local variance noise maps
- Context-guided feature fusion

### Example Comparison Models
Located in `models/`:
- `UNet.py`
- `unetr.py`
- `swin_unet.py`
- `WContextNet.py`
- `multiverseg/`

---

## Datasets

Supported ultrasound datasets:
- **BUSI** – Breast ultrasound
- **BUSBRA** – Breast ultrasound variant
- **CAMUS** – Cardiac ultrasound
- **JNU** – Fetal ultrasound with frame labels
- **Radboud-HC** – Fetal ultrasound training dataset

Data loading logic is implemented in `dataloaders.py`.

Expected structure example:
```
BUSI/
 ├── images/
 └── masks/
```

---

## Project Structure

```
ICL-Noise_UNet/
│
├── train_iclnoise.py        # Training + testing pipeline
├── dataloaders.py           # Dataset loading utilities
├── DataAugmentation.py      # Data augmentation
├── gpu.py                   # GPU helpers
├── models/
│   ├── NoiseContext.py      # ICL-Noise-UNet architecture
│   ├── UNet.py
│   ├── unetr.py
│   ├── swin_unet.py
│   └── WContextNet.py
└── requirements.txt
```

---

## Key Implementation Details

### train_iclnoise.py
- PyTorch Lightning `LightningModule`
- Custom `SoftDiceLoss`
- `TrainDataset` and `EvalDataset`
- Context sampling strategy
- Integrated testing after training



### NoiseContext.py
- ContextNoiseUNet class
- Noise-aware feature modulation
- Context fusion mechanisms

---

## Contributing

Contributions are welcome.
- Open issues for bugs or feature requests
- Submit pull requests with improvements or experiments

---

## License

MIT License. 
