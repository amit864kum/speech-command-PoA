# Blockchain-Enabled Federated Learning for Speech Commands

A comprehensive research platform combining federated learning, blockchain technology, and speech recognition for secure, decentralized machine learning on audio data.

## 🚀 Features

### Core Capabilities
- **Federated Learning**: Distributed training across multiple clients with privacy preservation
- **Blockchain Integration**: Secure model aggregation with multiple consensus mechanisms (PoW, PoA, PBFT)
- **Speech Recognition**: CNN-based models for Google Speech Commands dataset
- **Privacy Protection**: Differential privacy and robust aggregation methods
- **Decentralized Storage**: IPFS integration for off-chain model storage
- **Intelligent Optimization**: Reinforcement learning for adaptive parameter tuning

### Technical Highlights
- **Multiple Model Architectures**: SimpleAudioClassifier, GKWS_CNN, DS-CNN, ResNet
- **Data Distribution Options**: IID and non-IID data splits with Dirichlet distribution
- **Robust Aggregation**: Krum, Trimmed Mean, and FedAvg algorithms
- **Comprehensive Evaluation**: Privacy-utility tradeoffs, robustness testing, scalability analysis

## 📁 Project Structure

```
Speech_command/
├── src/                          # Core source code
│   ├── models/                   # ML model architectures
│   │   ├── simple_audio_classifier.py
│   │   ├── gkws_cnn.py
│   │   └── __init__.py
│   ├── data/                     # Data loading and preprocessing
│   │   ├── speech_commands_loader.py
│   │   ├── data_distribution.py
│   │   └── __init__.py
│   ├── federated/                # Federated learning components
│   ├── blockchain/               # Blockchain and consensus
│   ├── privacy/                  # Privacy-preserving mechanisms
│   ├── utils/                    # Utility functions
│   │   ├── config_loader.py
│   │   ├── logger.py
│   │   ├── reproducibility.py
│   │   └── __init__.py
│   └── __init__.py
├── configs/                      # Configuration files
│   └── default_config.yaml
├── scripts/                      # Experiment and utility scripts
├── tests/                        # Test suite
├── data/                         # Dataset storage
├── logs/                         # Experiment logs
├── results/                      # Experimental results
├── docs/                         # Documentation
├── requirements.txt              # Python dependencies
├── ENHANCEMENT_PLAN.md          # Development roadmap
└── README.md                    # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.12+
- CUDA (optional, for GPU acceleration)

### Setup
1. **Clone the repository**
   ```bash
   git clone https://github.com/amit864kum/Speech_command.git
   cd Speech_command
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download dataset** (automatic on first run)
   ```bash
   cd Speech_command
   python -c "from src.data import SpeechCommandsDataLoader; SpeechCommandsDataLoader(num_clients=3)"
   ```

## 🚀 Quick Start

### Basic Federated Learning Experiment
```python
from src.utils.config_loader import load_config
from src.utils.reproducibility import set_seed
from src.data import SpeechCommandsDataLoader
from src.models import SimpleAudioClassifier

# Load configuration
config = load_config()
set_seed(config.get('experiment.seed', 42))

# Initialize data loader
data_loader = SpeechCommandsDataLoader(
    num_clients=config.get('federated_learning.num_clients', 10),
    data_distribution=config.get('federated_learning.data_distribution', 'non_iid'),
    alpha=config.get('federated_learning.alpha', 0.5)
)

# Get client data
client_data = data_loader.get_client_data(client_id=0)
print(f"Client 0 has {len(client_data)} samples")

# Initialize model
model = SimpleAudioClassifier(
    input_dim=config.get('model.input_dim', 64),
    output_dim=config.get('model.output_dim', 10)
)
```

### Run Enhanced Experiment
```python
# Run the enhanced federated learning experiment
python src/main.py --config configs/default_config.yaml
```

## 📊 Configuration

The system uses YAML configuration files for easy experiment management. Key configuration sections:

### Dataset Configuration
```yaml
dataset:
  name: "SpeechCommands"
  version: "v0.02"
  target_words: ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
  data_distribution: "non_iid"  # "iid" or "non_iid"
  alpha: 0.5  # Dirichlet parameter for non-IID
```

### Federated Learning Parameters
```yaml
federated_learning:
  num_clients: 10
  num_rounds: 50
  local_epochs: 5
  aggregation_method: "fedavg"  # "fedavg", "krum", "trimmed_mean"
```

### Privacy Settings
```yaml
privacy:
  enable_differential_privacy: true
  epsilon: 1.0  # Privacy budget
  noise_multiplier: 1.1
```

### Blockchain Configuration
```yaml
blockchain:
  consensus_mechanism: "poa"  # "pow", "poa", "pbft"
  difficulty: 2
  mining_reward: 1.0
```

## 🔬 Research Features

### 1. Data Distribution Analysis
- **IID vs Non-IID**: Compare performance under different data distributions
- **Dirichlet Distribution**: Configurable non-IID scenarios with alpha parameter
- **Statistical Analysis**: Jensen-Shannon divergence for measuring non-IIDness

### 2. Privacy Mechanisms
- **Differential Privacy**: DP-SGD implementation with configurable privacy budgets
- **Secure Aggregation**: Cryptographic protocols for model aggregation
- **Privacy-Utility Tradeoffs**: Comprehensive evaluation of privacy vs accuracy

### 3. Robustness Testing
- **Byzantine Clients**: Simulation of malicious participants
- **Robust Aggregation**: Krum and Trimmed Mean algorithms
- **Attack Scenarios**: Model poisoning and inference attacks

### 4. Blockchain Integration
- **Multiple Consensus**: PoW, PoA, and PBFT implementations
- **Performance Metrics**: Block time, throughput, energy consumption
- **Decentralized Storage**: IPFS integration for model updates

## 📈 Experimental Evaluation

### Metrics Tracked
- **Model Performance**: Accuracy, loss, convergence rate
- **Privacy**: Privacy leakage, differential privacy guarantees
- **Robustness**: Attack success rate, Byzantine tolerance
- **Efficiency**: Communication overhead, computation time, energy consumption
- **Scalability**: Performance with varying number of clients

### Reproducibility
- **Deterministic Experiments**: Seed management for reproducible results
- **Configuration Management**: Version-controlled experiment settings
- **Comprehensive Logging**: Detailed experiment tracking and metrics

## 🔧 Development Roadmap

See [ENHANCEMENT_PLAN.md](ENHANCEMENT_PLAN.md) for detailed development phases:

1. **Phase 1**: ✅ Refactor & Setup (Current)
2. **Phase 2**: Federated Learning Core Enhancement
3. **Phase 3**: Robustness & Privacy Mechanisms
4. **Phase 4**: Advanced Blockchain Features
5. **Phase 5**: IPFS Integration
6. **Phase 6**: Reinforcement Learning Optimization
7. **Phase 7**: Comprehensive Experiments
8. **Phase 8**: Publication Artifacts

## 📚 Research Applications

### Potential Research Directions
- **Cross-lingual Speech Recognition**: Multi-language federated learning
- **Continual Learning**: Adapting to new commands without forgetting
- **Edge Computing**: Deployment on IoT and mobile devices
- **Privacy-Preserving Audio ML**: Advanced cryptographic techniques
- **Blockchain Scalability**: Consensus mechanisms for large-scale FL

### Publication Opportunities
- **Conferences**: ICML, NeurIPS, ICLR, AAAI, IJCAI
- **Journals**: IEEE TIFS, IEEE TMC, ACM TOPS, Computer Networks
- **Workshops**: FL-ICML, Blockchain-AI, Privacy-ML

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

- **Author**: Research Team
- **Email**: [your-email@domain.com]
- **GitHub**: [https://github.com/amit864kum/Speech_command](https://github.com/amit864kum/Speech_command)

## 🙏 Acknowledgments

- Google Speech Commands Dataset
- PyTorch and TorchAudio teams
- Federated Learning research community
- Blockchain and privacy-preserving ML researchers

## 📖 Citation

If you use this work in your research, please cite:

```bibtex
@misc{speech_command_fl_blockchain,
  title={Blockchain-Enabled Federated Learning for Speech Commands},
  author={Research Team},
  year={2024},
  url={https://github.com/amit864kum/Speech_command}
}
```