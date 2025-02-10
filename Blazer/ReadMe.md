# Autoencoder for Anomaly Detection in Blazar API Data

## Project Overview
This project trains an autoencoder using TensorFlow and Scikit-learn to detect anomalies in Blazar API data.

---
## Setup Guide

### 1. Install Python (if not installed)
- Download and install Python 3.10 or 3.11 from [Python.org](https://www.python.org/downloads/).
- Check "Add Python to PATH" during installation.

### 2. Create a Virtual Environment
Open a terminal and run:

```bash
python -m venv blazar_env  # Create virtual environment
```

### 3. Activate the Virtual Environment

#### Windows (Command Prompt / PowerShell):
```bash
blazar_env\Scripts\activate
```

#### macOS / Linux:
```bash
source blazar_env/bin/activate
```

---
## Install Dependencies
After activating the environment, install the required packages:

```bash
pip install --upgrade pip  # Upgrade pip
pip install -r requirements.txt  # Install dependencies
```

If you don’t have `requirements.txt`, create one:

```bash
echo "numpy\npandas\ntensorflow\nscikit-learn\nmatplotlib" > requirements.txt
pip install -r requirements.txt
```

---
## Run the Autoencoder Script
After setting up, run the main Python script:

```bash
python MainCode.py
```

If the script is in a different location, navigate there first:

```bash
cd path/to/script_directory
python MainCode.py
```

---
## Troubleshooting

### Virtual Environment Not Found?
Ensure you activated it before running any `pip` or `python` commands:
```bash
blazar_env\Scripts\activate  # Windows
source blazar_env/bin/activate  # macOS/Linux
```

### TensorFlow Installation Issues?
- Make sure you have Python 3.8 - 3.12 installed.
- If an error occurs, try:
  ```bash
  pip install --upgrade tensorflow
  ```

### Python Not Found?
Reinstall Python and check your PATH settings:
```bash
python --version
where python  # Windows
which python  # macOS/Linux
```
