#!/bin/bash
# TOMLCloud - Lambda Labs Instance Setup
# Run this ONCE after uploading the project to the instance.
# Usage: bash scripts/setup_instance.sh

set -e

echo "================================================"
echo "TOMLCloud - Lambda Labs Instance Setup"
echo "================================================"
echo "Started: $(date)"
echo ""

# Record start time for cost tracking
echo "Instance setup started at: $(date)" > data/cost_log.txt

# -----------------------------------------------------------
# Step 1: Verify GPU and power telemetry
# -----------------------------------------------------------
echo "[1/6] Verifying GPU and power telemetry..."
nvidia-smi
echo ""

POWER=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits | head -1)
if [[ "$POWER" == "[N/A]" || -z "$POWER" ]]; then
    echo "ERROR: Power readings return N/A. This instance cannot run TOMLCloud experiments."
    echo "Try a different instance or contact Lambda support."
    exit 1
fi
echo "Power reading OK: ${POWER}W"

nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit,clocks.max.sm,clocks.max.mem --format=csv > data/instance_metadata.csv
echo "GPU metadata saved to data/instance_metadata.csv"
echo ""

# -----------------------------------------------------------
# Step 2: Install Docker and NVIDIA Container Toolkit
# -----------------------------------------------------------
echo "[2/6] Installing Docker..."
sudo apt-get update -qq
sudo apt-get install -y -qq docker.io docker-compose > /dev/null 2>&1
sudo usermod -aG docker $USER
echo "Docker installed."

echo "[3/6] Installing NVIDIA Container Toolkit..."
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg 2>/dev/null
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
sudo apt-get update -qq
sudo apt-get install -y -qq nvidia-container-toolkit > /dev/null 2>&1
sudo nvidia-ctk runtime configure --runtime=docker > /dev/null 2>&1
sudo systemctl restart docker
echo "NVIDIA Container Toolkit installed."
echo ""

# -----------------------------------------------------------
# Step 3: Verify Docker GPU access
# -----------------------------------------------------------
echo "[4/6] Verifying Docker GPU access..."
sudo docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "Docker GPU access: OK"
else
    echo "WARNING: Docker GPU test failed. Containers may not have GPU access."
    echo "Continuing anyway - bare-metal experiments will still work."
fi
echo ""

# -----------------------------------------------------------
# Step 4: Build the TOMLCloud Docker image
# -----------------------------------------------------------
echo "[5/6] Building TOMLCloud Docker image..."
sudo docker build -t tomlcloud:latest -f docker/Dockerfile . 2>&1 | tail -5
echo "Docker image built."
echo ""

# -----------------------------------------------------------
# Step 5: Install Python dependencies for bare-metal runs
# -----------------------------------------------------------
echo "[6/6] Installing Python dependencies for bare-metal runs..."
pip install -q pynvml pandas numpy scipy torch torchvision xgboost scikit-learn 2>/dev/null
echo "Python dependencies installed."
echo ""

# -----------------------------------------------------------
# Record system info
# -----------------------------------------------------------
echo "--- System Info ---" >> data/instance_metadata.csv
uname -a >> data/instance_metadata.csv
cat /proc/cpuinfo | grep "model name" | head -1 >> data/instance_metadata.csv
free -h | head -2 >> data/instance_metadata.csv

echo "================================================"
echo "Setup complete at: $(date)"
echo "================================================"
echo ""
echo "Quick test (run this manually):"
echo "  cd ~/TOMLCloud"
echo "  python3 benchmarks/bench_resnet18.py --duration 5 --no-cooldown"
echo ""
echo "If that prints power/temp/clock readings, you are ready."
