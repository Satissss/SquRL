# SquRL

<div align="center">

**SquRL** - *A Scalable RL Framework for Training LLMs on Dynamic Text-to-SQL Workflow Construction.*

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

</div>

---

## 📖 Overview

**SquRL** (SQL Query Reinforcement Learning) is a state-of-the-art reinforcement learning framework designed for training large language models (LLMs) to generate high-quality SQL queries from natural language. Built on scalable distributed training techniques, SquRL enables efficient fine-tuning of LLMs using PPO (Proximal Policy Optimization) with custom reward signals tailored for Text-to-SQL tasks.


## Project Structure

```
SquRL/
├── scripts/                    # Training scripts
│   ├── rl_train_fsdp.sh       # RL training with FSDP
│   └── sft_peft_sp.sh         # SFT with LoRA + Sequence Parallel
├── verl/
│   ├── trainer/               # Training modules
│   │   ├── config/            # Hydra configuration files
│   │   ├── ppo/               # PPO trainer implementation
│   │   ├── main_ppo.py        # PPO training entry point
│   │   └── fsdp_sft_trainer.py # SFT trainer
│   ├── workers/               # Distributed workers
│   │   ├── actor/             # Actor model workers
│   │   ├── critic/            # Critic model workers
│   │   ├── rollout/           # Rollout generation workers
│   │   └── reward_model/      # Reward model workers
│   ├── utils/
│   │   ├── reward_score/      # Reward scoring functions
│   │   │   └── squrl.py       # SQL-specific reward scoring
│   │   └── dataset/           # Dataset utilities
│   └── third_party/
│       └── vllm/              # vLLM integration for multiple versions
├── patches/                   # Patches for external dependencies
├── requirements.txt           # Python dependencies
└── LICENSE                    # Apache 2.0 License
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- CUDA 11.8+ (for GPU training)
- Multiple GPUs recommended for distributed training
- Sufficient disk space for model checkpoints and datasets

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Satissss/SquRL.git
   cd SquRL
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   > **Note**: Make sure you have the correct CUDA version installed that matches your PyTorch installation.

3. **Setup Squrve Backend:**

   SquRL requires the Squrve backend for reward computation and evaluation.

   a. **Clone Squrve repository:**
   
   ```bash
   git clone https://github.com/Satissss/Squrve.git
   ```

   b. **Download the benchmark dataset:**
   
   Download from Hugging Face: [`satissss/Squrve-Benchmarks`](https://huggingface.co/datasets/satissss/Squrve-Benchmarks/tree/main)

   c. **Organize directory structure:**
   
   Place the `SquRL` directory under Squrve's `benchmarks/` folder:
   
   ```text
   Squrve/
   └── benchmarks/
       └── SquRL/
           ├── database/
           ├── rl/
           └── ...
   ```

   d. **Configure API keys:**
   
   ```bash
   cd Squrve/app
   vim app_config.json
   # Add your API KEY in the configuration file
   ```


## 📚 Training

### Step 1: Supervised Fine-Tuning (SFT)

First, train a base model using supervised fine-tuning with LoRA (Low-Rank Adaptation):

```bash
bash scripts/sft_peft_sp.sh
```

**Configuration:**
- Edit `scripts/sft_peft_sp.sh` to customize:
  - Model path and architecture
  - LoRA rank and alpha parameters
  - Learning rate and batch size
  - Sequence parallel settings

**Output:**
- Model checkpoints will be saved in the configured output directory
- Training logs and metrics are available in the logs folder

### Step 2: Reinforcement Learning Training (RL)

After SFT, continue with PPO-based reinforcement learning training:

1. **Start the Squrve backend server:**

   ```bash
   cd Squrve/app
   python run.py
   ```

   This starts the reward computation service for evaluating SQL query quality.

2. **Launch RL training:**

   ```bash
   cd SquRL
   bash scripts/rl_train_fsdp.sh
   ```


## 🤝 Contributing

Contributions are welcome! Please feel free to:

- Report bugs and issues
- Suggest new features or improvements
- Submit pull requests

Please ensure your code follows the existing style and includes appropriate tests.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on top of the [SQL-R1](https://github.com/DataArcTech/SQL-R1) framework
- Uses [Squrve](https://github.com/Satissss/Squrve) for Text-to-SQL evaluation

## 📞 Contact

For questions or support, please:
- Open an issue on GitHub
- Contact the maintainers

---

<div align="center">

**Star ⭐ this repository if you find it helpful!**

</div>
