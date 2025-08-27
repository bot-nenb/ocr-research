# OCR Batch Processing Demo

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Platform](https://img.shields.io/badge/Platform-Ubuntu%2024.04%20LTS-orange.svg)

A comprehensive high-volume OCR batch processing demonstration featuring **CPU-first architecture** with **GPU acceleration support**. This demo showcases cost-effective local OCR processing using EasyOCR and PaddleOCR with comprehensive performance monitoring, accuracy evaluation, and cost analysis.

## 🌟 Key Features

- **🖥️ CPU-First Design**: Optimized for accessibility without specialized hardware
- **⚡ GPU Acceleration Ready**: Seamless scaling when GPU resources become available  
- **📊 Comprehensive Monitoring**: Real-time CPU, memory, and GPU resource tracking
- **🎯 Accuracy Evaluation**: WER/CER analysis with statistical significance testing
- **💰 Cost-Benefit Analysis**: Local vs cloud processing cost comparisons
- **📈 Professional Reporting**: HTML reports, CSV exports, and executive summaries
- **🔧 Device-Aware Processing**: Automatic device detection and optimization
- **🛡️ Robust Error Handling**: Retry mechanisms and graceful failure recovery

## 📋 Table of Contents

- [System Requirements](#-system-requirements)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Data Setup](#-data-setup)
- [Usage Examples](#-usage-examples)
- [Configuration](#️-configuration)
- [Architecture](#-architecture)
- [Performance Benchmarks](#-performance-benchmarks)
- [Results and Reports](#-results-and-reports)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)

## 🖥️ System Requirements

### Minimum Requirements (CPU-Only Mode)
- **OS**: Ubuntu 24.04 LTS (recommended) or Ubuntu 20.04+
- **CPU**: 4-core processor (Intel/AMD)
- **Memory**: 8GB RAM
- **Storage**: 5GB free disk space
- **Python**: 3.10 or newer

### Recommended for GPU Acceleration
- **GPU**: NVIDIA GPU with 6GB+ VRAM
- **CUDA**: 11.8 or newer
- **Memory**: 16GB RAM
- **Storage**: 10GB free disk space

### Supported Operating Systems
- ✅ Ubuntu 24.04 LTS (Primary target)
- ✅ Ubuntu 22.04 LTS  
- ✅ Ubuntu 20.04 LTS
- ⚠️ Other Linux distributions (may require package adjustments)

## 🚀 Quick Start

Get up and running in under 5 minutes:

```bash
# 1. Clone and navigate to the project
cd tasks/task-1/demo/

# 2. Install uv package manager (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# 3. Set up the environment with CPU-only dependencies
uv sync --extra dev --extra cpu --extra paddle

# 4. Run the demo with default settings (processes 10 documents)
uv run python batch_processing_demo.py --num-documents 10 --config config.yaml

# 5. View results in the generated HTML report
firefox results/reports/*.html
```

## 📦 Installation

### Option 1: CPU-Only Setup (Recommended for Getting Started)

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Reload shell environment
source ~/.bashrc  # or restart your terminal

# Install CPU-optimized dependencies
uv sync --extra dev --extra cpu

# Verify installation
uv run python -c "import easyocr; print('✅ EasyOCR CPU setup complete')"
```

### Option 2: GPU-Accelerated Setup

```bash
# Ensure NVIDIA drivers and CUDA are installed
nvidia-smi  # Should show GPU information

# Install GPU-optimized dependencies  
uv sync --extra dev --extra cpu --extra gpu

# Verify GPU setup
uv run python -c "import torch; print(f'🚀 GPU available: {torch.cuda.is_available()}')"
```

## 📁 Data Setup

The demo uses the FUNSD (Form Understanding in Noisy Scanned Documents) dataset, which is **automatically downloaded** when you first run the demo. The `data/` directory is excluded from git to keep the repository lightweight.

### Automatic Data Download (Recommended)

```bash
# The dataset will be downloaded automatically on first run
uv run python batch_processing_demo.py --num-documents 10

# Or download the dataset separately
uv run python src/utils/dataset_loader.py
```

### Manual Data Setup (Optional)

If you prefer to download the dataset manually:

```bash
# Create data directory structure
mkdir -p data/funsd_subset/{images,annotations}

# Download FUNSD dataset
wget https://guillaumejaume.github.io/FUNSD/dataset.zip
unzip dataset.zip -d data/

# The demo will automatically organize and prepare the data
```

### Data Directory Structure

After setup, your data directory will look like:

```
data/
├── .gitkeep                    # Preserves directory in git
├── funsd_subset/               # Subset of FUNSD dataset
│   ├── images/                 # Document images (.png)
│   ├── annotations/            # Ground truth annotations (.json)
│   └── processed/              # Preprocessed data
└── funsd_original/             # Original FUNSD dataset (if downloaded)
```

### Data Requirements

- **Dataset Size**: ~500MB for full FUNSD dataset
- **Subset Size**: ~50-100MB for demo subset (100 documents)  
- **Format**: PNG images + JSON annotations
- **Languages**: English documents (forms and invoices)

### Offline Usage

Once downloaded, the demo works completely offline. The dataset and all processing are local, ensuring:
- ✅ **Privacy**: No data sent to external services
- ✅ **Speed**: No network latency for processing
- ✅ **Cost**: No API charges for OCR processing
- ✅ **Reliability**: Works without internet connection

## 💻 Usage Examples

### Basic Usage

```bash
# Process 50 documents with default settings
uv run python batch_processing_demo.py --num-documents 50

# Force CPU-only processing
uv run python batch_processing_demo.py --device cpu --num-documents 25

# Use custom configuration
uv run python batch_processing_demo.py --config configs/cpu_config.yaml
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--device` | Processing device: auto, cpu, gpu | auto |
| `--model` | OCR model: easyocr, paddleocr | easyocr |
| `--num-documents` | Number of documents to process | 100 |
| `--num-workers` | Number of parallel workers | auto-detect |
| `--batch-size` | Documents per batch | 10 |
| `--config` | YAML configuration file | None |
| `--output-dir` | Results output directory | results |
| `--log-level` | Logging verbosity | INFO |
| `--dry-run` | Test without processing | False |
| `--generate-reports` | Create HTML/CSV reports | True |

## 🏗️ Architecture

The system follows a modular, device-aware architecture:

```
Dataset → OCR Models → Batch Processor → Evaluation → Reports
    ↓         ↓            ↓              ↓           ↓
  FUNSD    EasyOCR    CPU/GPU Aware   WER/CER    HTML/CSV
```

### Key Components

1. **Dataset Loader**: FUNSD dataset preparation and validation
2. **OCR Models**: Device-aware EasyOCR wrapper
3. **Batch Processor**: Parallel processing with device optimization
4. **Monitoring**: Real-time resource tracking
5. **Evaluation**: Comprehensive accuracy analysis
6. **Cost Analysis**: Local vs cloud cost comparison
7. **Report Generation**: Professional reports and visualizations

## 📊 Performance Benchmarks

### Expected Performance (CPU Mode)

| Configuration | Docs/Minute | Memory Usage | Accuracy (WER) |
|---------------|-------------|--------------|----------------|
| 4-core CPU, 8GB RAM | 50-100 | <4GB | <0.15 |
| 8-core CPU, 16GB RAM | 100-200 | <6GB | <0.12 |

### Cost Analysis Results

Local processing typically achieves **60-85% cost savings** compared to cloud OCR APIs:

- **Google Vision API**: ~$1.50/1K documents
- **AWS Textract**: ~$1.50/1K documents  
- **Local Processing**: ~$0.05-0.15/1K documents

## 🔧 Troubleshooting

### Common Issues

**Problem**: `uv: command not found`
```bash
# Solution: Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

**Problem**: Out of memory errors
```bash
# Solution: Reduce batch size and worker count
uv run python batch_processing_demo.py --batch-size 4 --num-workers 2
```

**Problem**: GPU not detected
```bash
# Check NVIDIA setup
nvidia-smi
uv run python -c "import torch; print(torch.cuda.is_available())"
```

## 📈 Results and Reports

The demo generates comprehensive reports in the `results/` directory:

- **`complete_results.json`**: Full processing results and metrics
- **`reports/*.html`**: Interactive HTML reports with visualizations and **visual OCR comparison**
- **`executive_summary.json`**: High-level summary for stakeholders
- **`logs/`**: Detailed processing and error logs

### 🔍 Visual OCR Comparison (New Feature!)

The HTML reports now include an interactive visual comparison section featuring:

- **📸 Original Document Images**: View source documents alongside OCR results  
- **🎯 OCR Bounding Box Overlays**: See exactly where text was detected with confidence scores
- **📝 Side-by-Side Text Comparison**: Ground truth vs OCR output with highlighted differences:
  - 🔴 **Deletions**: Text missing from OCR output
  - 🟢 **Insertions**: Extra text detected by OCR  
  - 🟡 **Changes**: Text differences between ground truth and OCR

- **📊 Comparison Statistics**: Edit distance, similarity percentages, and word counts
- **🎛️ Interactive Controls**: Document selector and tabbed interface for easy navigation

Open any HTML report and navigate to the "🔍 Visual OCR Comparison" section to explore individual document results!

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository and create a feature branch
2. Follow code style with Black formatting
3. Add tests for new features
4. Update documentation
5. Submit pull request with detailed description

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **EasyOCR**: Excellent OCR library by JaidedAI
- **FUNSD Dataset**: Microsoft Research dataset for form understanding
- **PyTorch**: Deep learning framework powering EasyOCR

---

<div align="center">

**⭐ If this project helped you, please give it a star! ⭐**

</div>
