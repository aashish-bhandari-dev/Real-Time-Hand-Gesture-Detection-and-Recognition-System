# Contributing Guidelines

Thank you for contributing to the **Real-Time Hand Gesture Detection and Recognition System**!

## Development Setup

1. Clone your fork and create a branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Set up the development environment using Python 3.12:
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Fetch the MediaPipe model asset:
   ```bash
   python scripts/download_model.py
   ```

## Adding New Gestures

1. Open `src/config.py` and append the new gesture label to `GestureConfig.GESTURES`.
2. Launch the data collector to record training samples:
   ```bash
   python scripts/collect_data.py
   ```
   Select your new gesture with the number keys and record 150-250 diverse samples.
3. Retrain and evaluate the Random Forest model:
   ```bash
   python scripts/train_model.py
   python scripts/evaluate_model.py
   ```
4. Verify real-time detection in the main app:
   ```bash
   python main.py
   ```

## Code Quality Standards

Before committing your changes, ensure all tests pass and code style complies with Ruff:

```bash
# Run full offline test suite
pytest tests/ -v

# Lint and format checks
ruff check .
ruff format --check .
```
