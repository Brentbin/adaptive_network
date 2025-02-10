# Adaptive Network Project

This project implements an adaptive neural network system with enhanced thinking capabilities. The project is structured in multiple phases, each focusing on different aspects of the implementation.

## Project Structure

- `phase1/`: Initial implementation focusing on basic arithmetic and geometric operations
  - `models/`: Core network architecture and adaptive components
  - `experiments/`: Training scripts and experimental results

- `phase2/`: Enhanced implementation with advanced thinking capabilities
  - `models/enhanced_thinking/`: Advanced network components
    - Communication system
    - Performance monitoring
    - Tensor utilities
    - System management
  - `tests/`: Comprehensive test framework
  - `docs/`: Documentation and experiment tracking

- `docs/`: Project-wide documentation
  - Study notes
  - Related work references

## Requirements

- Python 3.9+
- PyTorch
- NumPy
- Additional dependencies can be found in `requirements.txt`

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/Brentbin/adaptive_network.git
cd adaptive_network
```

2. Create and activate a virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows use: env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Phases

### Phase 1
Initial implementation focusing on fundamental operations and basic adaptive capabilities. See `phase1/README.md` for details.

### Phase 2
Enhanced implementation with advanced thinking capabilities, improved performance monitoring, and comprehensive testing framework.

## Testing

Run the test suite:
```bash
cd phase2/tests
python run_tests.py
```

## Documentation

Detailed documentation for each phase can be found in their respective directories:
- Phase 1: `phase1/README.md`
- Phase 2: `phase2/docs/`

## License

MIT License 