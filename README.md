# TuneForge
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**TuneForge** is a lightweight, flexible tool for fine-tuning large language models (LLMs) on any text dataset. Powered by [Unsloth](https://github.com/unslothai/unsloth) and LoRA, it supports efficient 4-bit quantization, making it ideal for GPUs with 24GB VRAM or less. Whether you're generating game levels, crafting custom text, or adapting models to unique tasks, TuneForge has you covered.

- **Fine-tune**: Train models like Qwen2.5-7B, Gemma-3 (4B/12B), LLaMA-3.3-8B, and more.
- **Generate**: Complete prompts with your fine-tuned models.
- **Convert**: Export to GGUF format for lightweight deployment.

Tested on a 24GB RTX 4090—scales to 7B–12B models with ease!

---

## Features
- **Multi-Model Support**: Fine-tune Qwen, Gemma-3, LLaMA, DeepSeek, and more.
- **Flexible Training**: Choose epochs and normal/low-memory modes (batch size 8 or 4).
- **Efficient**: Uses 4-bit quantization and gradient checkpointing for low VRAM usage.
- **Output Options**: Generate text or convert models to GGUF via llama.cpp.
- **User-Friendly**: Interactive CLI with defaults for quick setup.

---

## Installation

### Prerequisites
- **Python**: 3.10 or 3.12 (recommended).
- **GPU**: NVIDIA GPU with CUDA support (e.g., RTX 3090, 4090). 24GB VRAM recommended for 12B models.
- **OS**: Linux (Ubuntu tested), Windows (WSL2 recommended), or macOS (limited GPU support).
- **Disk Space**: ~20GB+ for models and dependencies.

#### 1. Clone the Repository
```bash
git clone https://github.com/vedran/tuneforge.git
cd tuneforge
```

#### 2. Set Up a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

#### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install unsloth transformers datasets trl
```

- **Linux/Windows Notes**: Use `cu121` for CUDA 12.1 (adjust for your CUDA version: `nvidia-smi`).
- **macOS Notes**: Replace `cu121` with `cpu` (no GPU support yet with Unsloth).

#### 4. (Optional) GGUF Conversion
For GGUF export, install [llama.cpp](https://github.com/ggerganov/llama.cpp):
```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
make
cd ..
```
Ensure `llama.cpp/convert-hf-to-gguf.py` is accessible in the `tuneforge` directory.

---

## Platform-Specific Setup

### Linux (Ubuntu 22.04+)
1. **Install NVIDIA Drivers & CUDA**:
   ```bash
   sudo apt update
   sudo apt install nvidia-driver-550 nvidia-utils-550 cuda-12-2
   ```
2. **Verify**:
   ```bash
   nvidia-smi  # Should show GPU info
   nvcc --version  # Should show CUDA 12.2
   ```
3. Follow general installation steps above.

### Windows (via WSL2)
1. **Set Up WSL2**:
   - Install WSL2 and Ubuntu (e.g., `wsl --install` in PowerShell, then `Ubuntu-22.04` from Microsoft Store).
   - Install NVIDIA CUDA drivers for WSL2: [NVIDIA Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).
2. **In WSL2 Ubuntu**:
   ```bash
   sudo apt update
   sudo apt install python3-pip python3-venv
   ```
3. Follow general installation steps, adjusting paths (e.g., `venv/bin/activate`).

### macOS
- **Limitations**: No GPU acceleration with Unsloth (CPU-only).
- **Install**:
  ```bash
  brew install python
  ```
- Use `pip install torch torchvision` (CPU version) and follow general steps.

---

## Usage

### Prepare Your Dataset
- Create a text file (e.g., `swampattack2.txt`) with your training data.
- Format: Entries separated by `-----\n` (e.g., one per line or block).
- Example:
  ```
  Entry 1
  -----
  Entry 2
  -----
  Entry 3
  ```

### Run TuneForge
```bash
python tuneforge.py
```

#### Menu Options
1. **Fine-tune a Model**:
   - Select a model (e.g., `gemma-3-12b-pt`).
   - Enter epochs (default: 10).
   - Choose mode:
     - **Normal**: Batch size 8 (7B or smaller, 24GB VRAM).
     - **Lowmem**: Batch size 4 (12B models or low VRAM).
   - Output saved to `./swamp_model_<model>/final`.

2. **Complete Prompt**:
   - Uses the fine-tuned model (or base if not fine-tuned).
   - Generates text from a default prompt (edit `PROMPT_TEMPLATE` in script).
   - Saves to `generated_level_<model>.txt`.

3. **Convert to GGUF**:
   - Converts fine-tuned model to GGUF format (requires llama.cpp).
   - Output: `<model>.gguf`.

4. **Exit**: Closes the tool.

#### Example Run
```
=== TuneForge LLM ===
1. Fine-tune a model
2. Complete prompt with finetuned model
3. Convert to GGUF
4. Exit
Select an option (1-4): 1

Available models:
  a. qwen2.5-7b
  e. gemma-3-12b-pt
Select a model by letter (a-g): e

Fine-tuning gemma-3-12b-pt...
Enter number of epochs (default 10): 12
Training mode options:
  1. Normal: Batch size 8 (2×4)
  2. Lowmem: Batch size 4 (1×4)
  (Recommended: Lowmem for 12B models like gemma-3-12b-pt)
Select mode (1 for Normal, 2 for Lowmem): 2
```

---

## Guidelines

### Dataset Tips
- **Size**: 100–1000 samples work well (e.g., 410 samples tested).
- **Format**: Consistent separators (`-----\n`) ensure proper splitting.

### Training Recommendations
- **Epochs**: 5–15 (10 default). Monitor loss in logs:
  - Early plateau (<0.4): Reduce to 5–7.
  - Still dropping (<0.38 at 10): Try 12–15.
- **Learning Rate**: 2e-5 (default). Increase to 5e-5 if slow.
- **Mode**:
  - **Normal**: 7B models (e.g., DeepSeek-R1-7B) on 24GB VRAM.
  - **Lowmem**: 12B models (e.g., Gemma-3-12B) or if VRAM maxes out.

### Hardware
- **24GB VRAM**: Handles 7B (normal) and 12B (lowmem) models.
- **Monitor**: Use `nvidia-smi` to check VRAM usage.

---

## Troubleshooting

### CUDA Errors
- **"CUDA driver error: unknown error"**:
  - Reset GPU: `sudo nvidia-smi --gpu-reset`.
  - Update drivers/CUDA: See Linux/Windows setup.
  - Reduce `max_seq_length` to 1024 if VRAM spikes.

### File Descriptor Limit
- **"Too many open files"**:
  - Increase limit:
    ```bash
    ulimit -n 4096  # Temporary
    ```
    Edit `/etc/security/limits.conf` for permanence:
    ```
    * soft nofile 4096
    * hard nofile 4096
    ```

### Model-Specific Issues
- **Gemma-3 Errors**: Ensure Unsloth is updated:
  ```bash
  pip install --upgrade unsloth
  ```
- **VRAM Overload**: Switch to lowmem mode or lower `max_seq_length`.

### Logs
- Check `<output_dir>/training.log` for loss/gradient norms to diagnose convergence.

---

## Contributing
- Fork the repo, tweak `tuneforge.py`, and submit a PR!
- Ideas: Add more models, custom LR prompts, or dataset format options.

## License
MIT License—free to use, modify, and share. See [LICENSE](LICENSE).

## Acknowledgments
- Built with [Unsloth](https://github.com/unslothai/unsloth) for fast fine-tuning.
- Thanks to the AI and open-source community for inspiration!

Happy tuning with TuneForge! Questions? Open an issue or ping me on GitHub.
