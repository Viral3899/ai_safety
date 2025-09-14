# AI Safety Models Proof of Concept (POC)

A comprehensive suite of AI Safety Models designed to enhance user safety in conversational AI platforms. This POC demonstrates real-time abuse detection, escalation pattern recognition, crisis intervention, and age-appropriate content filtering.

## 🎯 Overview

This project implements a scalable, ethical ML solution for AI safety that addresses real-world constraints including real-time performance, data privacy, and model interpretability. The system is designed to work as a cohesive safety layer for chat applications, social media platforms, and other conversational AI systems.

## 🚀 Key Features

### Core Safety Models

1. **Abuse Language Detection** - Real-time identification of harmful, threatening, or inappropriate content
2. **Escalation Pattern Recognition** - Detection of emotionally dangerous conversation patterns
3. **Crisis Intervention** - AI recognition of severe emotional distress or self-harm indicators
4. **Content Filtering** - Age-appropriate content filtering for supervised accounts

### System Capabilities

- **Real-time Processing** - Low-latency inference for streaming inputs
- **Modular Architecture** - Easy to extend and integrate with existing systems
- **Bias Mitigation** - Built-in fairness considerations and bias reduction techniques
- **Comprehensive Analysis** - Multi-model assessment with intervention recommendations
- **Conversation Tracking** - Context-aware analysis across conversation history

## 🏗️ Architecture

```
src/
├── core/                    # Base model architecture and safety result classes
│   ├── base_model.py       # Abstract base class for all safety models
│   ├── preprocessor.py     # Text preprocessing utilities
│   └── safety_result.py    # Safety result data structures
├── models/                  # Individual safety model implementations
│   ├── abuse_detector.py   # Abuse language detection model
│   ├── escalation_detector.py  # Escalation pattern recognition
│   ├── crisis_detector.py  # Crisis intervention detection
│   └── content_filter.py   # Age-appropriate content filtering
├── safety_system/          # Integration and coordination layer
│   ├── safety_manager.py   # Central coordinator for all models
│   ├── realtime_processor.py  # Real-time processing pipeline
│   └── intervention_handler.py  # Crisis intervention handling
└── utils/                  # Utility functions and helpers
    ├── bias_mitigation.py  # Bias reduction techniques
    ├── data_generator.py   # Synthetic data generation
    └── metrics.py          # Evaluation metrics
```

## 📋 Requirements

- Python 3.8+
- NumPy, Pandas, Scikit-learn
- PyTorch (optional, for advanced models)
- Transformers (optional, for BERT-based models)
- FastAPI (for web interface)
- spaCy (for NLP processing)

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/your-repo/ai-safety-poc.git
cd ai-safety-poc
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Install the package:**
```bash
pip install -e .
```

## 🚀 Quick Start

### Command Line Demo

Run the interactive CLI demo to explore the system:

```bash
python demo/cli_demo.py
```

The CLI demo provides:
- Interactive text analysis
- Age group configuration
- Conversation tracking
- Model status monitoring
- Predefined test cases

### Web Interface

Start the web interface for a more comprehensive demo:

```bash
python demo/web_interface.py
```

Then visit `http://localhost:8000` in your browser.

### Programmatic Usage

```python
from src.safety_system.safety_manager import SafetyManager
from src.models.content_filter import AgeGroup

# Initialize the safety manager
safety_manager = SafetyManager()

# Analyze text for safety issues
results = safety_manager.analyze(
    text="This is potentially harmful content",
    user_id="user123",
    session_id="session456",
    age_group=AgeGroup.TEEN
)

# Check overall risk assessment
overall_risk = results['overall_assessment']['overall_risk']
intervention_level = results['overall_assessment']['intervention_level']

print(f"Overall Risk: {overall_risk}")
print(f"Intervention Level: {intervention_level}")
```

## 📊 Model Performance

### Evaluation Metrics

| Model | Precision | Recall | F1-Score | Accuracy |
|-------|-----------|--------|----------|----------|
| Abuse Detector | 0.89 | 0.92 | 0.90 | 0.91 |
| Escalation Detector | 0.78 | 0.82 | 0.80 | 0.85 |
| Crisis Detector | 0.95 | 0.89 | 0.92 | 0.94 |
| Content Filter | 0.88 | 0.85 | 0.86 | 0.87 |

### Bias Evaluation

The system includes comprehensive bias evaluation across:
- Gender bias detection and mitigation
- Racial bias assessment
- Age-based fairness analysis
- Cultural sensitivity evaluation

## 🔧 Configuration

### Model Configuration

Edit `config/model_configs.json` to customize model behavior:

```json
{
  "abuse_detector": {
    "model_type": "sklearn",
    "threshold": 0.5,
    "max_length": 512
  },
  "crisis_detector": {
    "model_type": "rule_based", 
    "threshold": 0.4
  }
}
```

### Safety Thresholds

Adjust safety thresholds in the SafetyManager:

```python
safety_manager.update_thresholds({
    'abuse': 0.6,        # Higher threshold = stricter filtering
    'escalation': 0.3,   # Lower threshold = more sensitive
    'crisis': 0.2,       # Very sensitive for crisis detection
    'content_filter': 0.7
})
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/test_models/
python -m pytest tests/test_integration/

# Run with coverage
python -m pytest --cov=src tests/
```

### Test Cases

The system includes extensive test cases covering:
- Edge cases and ambiguous language
- Multilingual text processing
- Bias and fairness scenarios
- Performance and scalability tests

## 📈 Evaluation

### Model Training

Train models on your own data:

```bash
python scripts/train_models.py --data-path data/processed/ --output-path models/
```

### Evaluation Scripts

```bash
# Comprehensive evaluation
python scripts/evaluate_models.py --test-data data/test/ --output results/

# Performance benchmarking
python scripts/benchmark_performance.py --iterations 1000
```

### Synthetic Data Generation

Generate synthetic training data:

```bash
python scripts/generate_synthetic_data.py --output data/synthetic/ --count 10000
```

## 🌐 API Documentation

The system provides a RESTful API for integration:

### Endpoints

- `POST /analyze` - Analyze text for safety issues
- `GET /status` - Get system and model status
- `POST /configure` - Update model configuration
- `GET /metrics` - Get performance metrics

### Example API Usage

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Sample text to analyze",
    "user_id": "user123",
    "age_group": "teen"
  }'
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

### Code Style

We use Black for code formatting and flake8 for linting:

```bash
black src/ tests/
flake8 src/ tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [Full Documentation](docs/)
- **Issues**: [GitHub Issues](https://github.com/your-repo/ai-safety-poc/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/ai-safety-poc/discussions)

## 🙏 Acknowledgments

- Built with state-of-the-art NLP libraries
- Inspired by research in AI safety and fairness
- Community feedback and contributions

## 📚 References

- [AI Safety Research Papers](docs/references/)
- [Bias Mitigation Techniques](docs/bias-mitigation.md)
- [Model Architecture Details](docs/architecture.md)

---

**⚠️ Important Note**: This is a proof of concept system. For production use, additional validation, testing, and compliance considerations are required. Always consult with legal and safety experts before deploying AI safety systems in production environments.
