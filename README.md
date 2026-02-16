# MLOps Sample Project: MNIST Digit Classification

This project demonstrates **Machine Learning Operations (MLOps)** best practices using a simple Convolutional Neural Network (CNN) for MNIST digit classification. It's designed as a learning resource to understand how to structure, train, evaluate, and deploy ML models in a production-ready manner.

## 🎯 What is MLOps?

MLOps (Machine Learning Operations) is a set of practices that combines Machine Learning, DevOps, and Data Engineering to deploy and maintain ML systems in production reliably and efficiently. This project covers:

- **Configuration Management**: Centralized config files for reproducibility
- **Model Versioning**: Tracking different model versions and checkpoints
- **Logging & Monitoring**: TensorBoard integration for training metrics
- **Model Checkpointing**: Saving models during training
- **Evaluation Pipeline**: Systematic model evaluation with metrics
- **Inference Pipeline**: Making predictions on new data
- **Model Serving API**: REST API with FastAPI for inference
- **Project Structure**: Organized codebase following best practices

## 📁 Project Structure

```
.
├── .github/workflows/
│   └── ci.yml                # GitHub Actions CI
├── configs/
│   └── config.yaml           # Centralized configuration
├── src/
│   ├── app.py                # FastAPI model server
│   ├── model.py              # Model architecture
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation script
│   └── inference.py          # Inference script
├── tests/                    # Pytest tests
├── models/                   # Saved models and checkpoints
├── logs/                     # Training logs and visualizations
├── data/                     # Dataset (auto-downloaded)
├── requirements.txt         # Python dependencies
└── README.md                 # This file
```

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Your Setup

Edit `configs/config.yaml` to adjust hyperparameters:
- Batch size
- Learning rate
- Number of epochs
- Device (CPU/GPU)

### 3. Train the Model

From the project root directory, run:

```bash
python src/train.py
```

**Note**: Make sure you're in the project root directory (where `README.md` is located) when running the scripts.

This will:
- Download the MNIST dataset automatically
- Train the CNN model
- Save checkpoints during training
- Log metrics to TensorBoard
- Save the best model based on validation accuracy

**View training progress with TensorBoard:**
```bash
tensorboard --logdir logs
```
Then open http://localhost:6006 in your browser.

### 4. Evaluate the Model

From the project root directory:

```bash
python src/evaluate.py
```

This will:
- Load the best trained model
- Evaluate on the test set
- Generate a classification report
- Create a confusion matrix visualization

### 5. Make Predictions

From the project root directory:

```bash
python src/inference.py --image path/to/your/image.png
```

**Example**: If you have a test image, you can use it like:
```bash
python src/inference.py --image test_image.png
```

### 6. Run Tests

```bash
pip install pytest
pytest tests/ -v
```

### 7. Serve the Model with FastAPI

From the project root (after training a model):

```bash
uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service info and links |
| `/health` | GET | Health check; verifies model is loaded |
| `/predict` | POST | Upload image; returns predicted digit and probabilities |
| `/docs` | GET | Interactive Swagger UI |

**Predict** accepts a multipart form with `file` = image (PNG, JPEG, etc.):

```bash
curl -X POST http://localhost:8000/predict -F "file=@path/to/digit.png"
```

Response:
```json
{"predicted_digit": 7, "confidence": 0.991234, "probabilities": {"0": 0.001, "1": 0.002, ...}}
```

The API loads `models/checkpoints/best.pth` (or `final_model.pth`) at startup and uses the same preprocessing as training.

## 📊 Model Architecture

The model is a simple CNN with:
- **Conv Layer 1**: 32 filters, 3x3 kernel
- **Conv Layer 2**: 64 filters, 3x3 kernel
- **Fully Connected**: 128 neurons → 10 output classes
- **Dropout**: 0.25 for regularization

## 🔧 Key MLOps Concepts Demonstrated

### 1. **Configuration Management**
All hyperparameters are stored in `configs/config.yaml`. This ensures:
- Reproducibility: Same config = same results
- Easy experimentation: Change config without touching code
- Version control: Track config changes in git

### 2. **Model Versioning**
- Models are saved with timestamps
- Best model is tracked separately
- Checkpoints include epoch, optimizer state, and metrics

### 3. **Logging & Monitoring**
- TensorBoard integration for real-time metrics
- Training/validation loss and accuracy tracking
- Config saved with each training run

### 4. **Data Pipeline**
- Consistent preprocessing (normalization)
- Train/validation split
- Reproducible data loading

### 5. **Evaluation Pipeline**
- Systematic test set evaluation
- Multiple metrics (accuracy, classification report)
- Visualization of results (confusion matrix)

### 6. **Inference Pipeline**
- Consistent preprocessing for new data
- Confidence scores
- Visualization of predictions

## 📈 Expected Results

After training (default: 15 epochs in config), you should see:
- **Training Accuracy**: ~98-99%
- **Validation Accuracy**: ~98-99%
- **Test Accuracy**: ~98-99%

## 🎓 Learning Path

1. **Start Simple**: Run the training script and observe the output
2. **Experiment**: Modify hyperparameters in `config.yaml`
3. **Monitor**: Use TensorBoard to visualize training
4. **Evaluate**: Run evaluation to see detailed metrics
5. **Extend**: Try adding:
   - Data augmentation
   - Different architectures
   - Early stopping
   - Learning rate scheduling
   - Model serving API (included via FastAPI)

## 🔍 Understanding the Code

### `src/model.py`
Defines the neural network architecture. This is where you'd experiment with different model designs.

### `src/train.py`
The training pipeline. Key features:
- Loads config
- Sets up data loaders
- Training loop with logging
- Checkpoint saving
- TensorBoard integration

### `src/evaluate.py`
Evaluation pipeline:
- Loads trained model
- Tests on held-out test set
- Generates metrics and visualizations

### `src/inference.py`
CLI inference:
- Loads model
- Preprocesses new images
- Makes predictions
- Provides confidence scores

### `src/app.py`
FastAPI REST server:
- Loads model once at startup
- `POST /predict` accepts image uploads and returns digit + probabilities
- Same preprocessing as training/inference for consistency

## 🔄 CI (GitHub Actions)

CI runs on every push and pull request to `main` or `master`:

- **Workflow**: `.github/workflows/ci.yml`
- **Jobs**: Install dependencies, run `pytest tests/` on Python 3.10, 3.11, and 3.12
- **Tests**: Model creation/forward pass and config validation in `tests/`

No secrets or deployment steps are required; push your branch and check the **Actions** tab on GitHub.

## 🛠️ Next Steps for Advanced MLOps

1. **Model Registry**: Use MLflow or Weights & Biases
2. **Containerization**: Dockerize the application (include FastAPI server)
3. **Monitoring**: Track model drift and performance in production
4. **A/B Testing**: Compare different model versions
5. **Data Versioning**: Track dataset versions with DVC

## 📝 Notes

- The dataset is automatically downloaded on first run
- Models are saved in `models/` directory
- Logs are saved in `logs/` directory
- Use GPU if available by setting `device: "cuda"` in config

## 🤝 Contributing

Feel free to experiment and extend this project! Some ideas:
- Add data augmentation
- Implement different architectures
- Add model serving API
- Integrate with MLflow or W&B
- Add unit tests

## 📚 Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [MLOps Guide](https://ml-ops.org/)
- [TensorBoard Tutorial](https://www.tensorflow.org/tensorboard)

## 🐛 Troubleshooting

**Issue**: CUDA out of memory
- **Solution**: Reduce batch size in `config.yaml`

**Issue**: Module not found
- **Solution**: Install dependencies with `pip install -r requirements.txt`

**Issue**: Model file not found
- **Solution**: Run `train.py` first to train a model

**Issue**: FastAPI server fails to start (model not found)
- **Solution**: Ensure you've run `python src/train.py` and that `models/checkpoints/best.pth` or `models/final_model.pth` exists

---

Happy Learning! 🎉
