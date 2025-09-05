# 3D Print Quality Control System

This repository contains a highly customizable deep learning system for 3D print quality control. The system uses a dual-backbone neural network to compare print images against reference images and classify them as good or bad prints.

## Features

- Customizable model architecture with different backbones (ResNet18, ResNet34, ResNet50, EfficientNet, MobileNet)
- Configurable similarity and classification heads
- Extensive data augmentation options
- Multiple loss function options
- Various optimizers and learning rate schedulers
- Comprehensive logging and visualization with TensorBoard
- Early stopping and model checkpointing
- Stratified data splitting for balanced training
- Batch inference on test data
- Visual result analysis

## Directory Structure

```
.
├── config_loader.py    # Configuration loading utilities
├── data_utils.py       # Dataset and data loading utilities
├── model.py            # Model architecture definitions
├── training.py         # Training and logging utilities
├── main.py             # Main training script
├── inference.py        # Inference script
└── config.json         # Configuration file
```

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install torch torchvision albumentations tensorboard matplotlib seaborn scikit-learn pillow
```

## Data Preparation

Organize your data in the following structure:

```
data/
├── Bad_png/
│   ├── ID_1.png
│   ├── ID_3.png
│   └── ...
├── Good_png/
│   ├──ID_6
│   ├──ID_9
│   └── ...
└── IDCBR_reference.png
└── IDCR_reference.png
└── IDSQR_reference.png
└── IDSR_reference.png
```
## 3D Bioprnting Features

```
All bioprinting parameter features are available in the 3D bioprinting features Excel file.
```

## Configuration

The system is highly customizable through the `config.json` file. The file is divided into several sections:

1. **General Settings**: Controls the overall experiment parameters
2. **Data Settings**: Parameters for data loading and preprocessing
3. **Model Settings**: Model architecture configuration
4. **Training Settings**: Training process parameters
5. **Optimizer Settings**: Optimizer configuration
6. **Scheduler Settings**: Learning rate scheduler configuration
7. **Early Stopping**: Early stopping parameters
8. **Augmentation Settings**: Data augmentation configuration
9. **Logging Settings**: Logging and checkpoint configuration

Edit the `config.json` file to customize the system according to your needs.

## Training

To train the model with default configuration:

```bash
python main.py --config config.json
```

You can override specific configuration parameters via command line:

```bash
python main.py --config config.json --data_root /path/to/data --epochs 150 --batch_size 64 --lr 0.0005
```

The training script will:
1. Load and validate the configuration
2. Prepare the data loaders
3. Initialize the model, optimizer, and scheduler
4. Train the model for the specified number of epochs
5. Validate the model after each epoch
6. Save checkpoints and logs
7. Generate visualizations of training progress

## Inference

To run inference on a single image:

```bash
python inference.py --model path/to/model.pt --image path/to/image.png --reference path/to/reference.png --visualize
```

To run batch inference on a test dataset:

```bash
python inference.py --model path/to/model.pt --data_dir path/to/data_directory --output_dir path/to/output_directory --visualize
```

## Experiment Tracking

The system uses TensorBoard for experiment tracking. To view the training progress:

```bash
tensorboard --logdir experiments/
```

This will allow you to visualize:
- Training and validation losses
- Classification and similarity component losses
- Validation accuracy
- Learning rate changes
- Confusion matrices
- Classification reports

## Examples

### Custom Model Configuration

To use a different backbone with custom head layers:

```json
{
    "model_architecture": "QualityControlNet",
    "backbone": "resnet50",
    "pretrained": true,
    "dropout_rate": 0.6,
    "similarity_head_layers": [2048, 1024, 512, 256, 1],
    "classification_head_layers": [2048, 1024, 512, 256, 2]
}
```

### Learning Rate Schedule

To use cosine annealing with warm restarts:

```json
{
    "scheduler": "CosineAnnealingWarmRestarts",
    "scheduler_T_0": 10,
    "scheduler_T_mult": 2,
    "scheduler_min_lr": 1e-6
}
```

### Custom Augmentation

To focus on specific augmentations:

```json
{
    "use_augmentation": true,
    "aug_rotate90_prob": 0.7,
    "aug_flip_prob": 0.7,
    "aug_shift_scale_rotate_prob": 0.6,
    "aug_noise_prob": 0.3,
    "aug_blur_prob": 0.1,
    "aug_color_prob": 0.4
}
```

## Advanced Usage

### Custom Loss Weighting

Adjust the weights of the classification and similarity losses:

```json
{
    "classification_weight": 0.7,
    "similarity_weight": 1.3
}
```

### Early Stopping Customization

Configure early stopping to be more or less aggressive:

```json
{
    "early_stopping": true,
    "early_stopping_patience": 20,
    "early_stopping_min_delta": 0.0005
}
```

## Performance Tips

1. **GPU Acceleration**: Set `"device": "cuda"` and `"pin_memory": true` for faster training on GPU
2. **Data Loading**: Adjust `"num_workers"` based on your CPU cores (typically 4-8)
3. **Batch Size**: Use the largest batch size that fits in your GPU memory
4. **Augmentation**: More augmentation helps with limited data
5. **Learning Rate**: Start with 0.001 and adjust based on training curves
6. **Early Stopping**: Use early stopping to prevent overfitting
7. **Model Size**: Smaller backbones (MobileNet) train faster but may be less accurate

## Extending the System

The modular design makes it easy to extend the system:

1. **New Backbones**: Add new backbone models in `model.py`
2. **Custom Losses**: Add new loss functions in `training.py`
3. **Additional Metrics**: Add new metrics in `ExperimentLogger.log_metrics()`
4. **Custom Schedulers**: Add new schedulers in `get_scheduler()`
