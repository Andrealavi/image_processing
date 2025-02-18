# Digital Image Processing Examples

This repository contains a collection of digital image processing implementations and examples. The code is written in Python, leveraging popular scientific computing libraries.

## Table of Contents
- [Overview](#overview)
- [Current Implementations](#current-implementations)
- [Theoretical Background](#theoretical-background)
- [Requirements](#requirements)
- [Usage](#usage)

## Overview

This repository serves as a practical resource for digital image processing techniques. Each implementation is accompanied by theoretical explanations and practical examples. The repository is structured to accommodate future additions of various image processing topics.

## Current Implementations

### Filtering Techniques
- **Mean Filters**: Various implementations including arithmetic, geometric, harmonic, and contraharmonic means
- **Median Filter**: Non-linear filtering approach
- **Gaussian Filter**: Linear smoothing filter

### Edge Detection
- **Canny Edge Detection**: Complete implementation including:
  - Gaussian smoothing
  - Gradient computation
  - Non-maximum suppression
  - Double thresholding and edge tracking

### Noise Treatment
- **Periodic Noise**: Basic handling of periodic noise patterns
- **Salt and Pepper Noise**: Removal using specialized filters
- **Gaussian Noise**: Treatment using appropriate filtering techniques

## Theoretical Background

Each implementation includes relevant theoretical foundations in its corresponding documentation. The code is thoroughly commented to explain the mathematical concepts being applied.

Some key concepts currently covered include:

### Filtering Operations
Digital filters are neighborhood operations where pixel values are modified based on surrounding pixels. Different filtering approaches are suitable for different types of image processing tasks.

### Edge Detection
Edge detection identifies significant local changes in image intensity, crucial for feature detection and image segmentation.

### Noise Reduction
Various noise reduction techniques are implemented based on different noise models and their statistical properties.

## Requirements and Setup

### Prerequisites
- Python 3.x
- pip (Python package installer)

### Environment Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:

On Windows:
```bash
venv\Scripts\activate
```

On Unix or MacOS:
```bash
source venv/bin/activate
```

4. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

Each script can be run independently. Example images are provided in the `img` directory.

Basic usage:
```bash
python src/filters/mean_filters.py
```

For detailed usage of specific implementations, refer to the documentation in each module.
