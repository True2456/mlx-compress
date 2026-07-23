#!/usr/bin/env bash
# ==============================================================================
# GCP Phase 2-4: GPU REAP Saliency, Gated Ladder, & HF Upload Pipeline
# ==============================================================================
# Project: project-e60cdf5e-8d01-4a57-9f0
# Region: us-central1
# VM: 8x H100 80GB (a3-highgpu-8g) or 8x A100 80GB (a2-highgpu-8g)
# Disk: Attached 1.5 TB SSD (step37-disk)
# ==============================================================================

set -euo pipefail

GCLOUD_BIN="$(which gcloud 2>/dev/null || echo "/Users/true/google-cloud-sdk/bin/gcloud")"
PROJECT_ID="project-e60cdf5e-8d01-4a57-9f0"
ZONE="us-central1-a"
GPU_VM_NAME="step37-gpu-reap"
DISK_NAME="step37-disk"

echo "======================================================================="
echo "PHASE 2-4: Provisioning REAP GPU Compute VM & Attaching Disk"
echo "======================================================================="

# 1. Provision GPU VM with attached 1.5 TB disk
echo "⚡ Provisioning GPU VM (${GPU_VM_NAME})..."
${GCLOUD_BIN} compute instances create ${GPU_VM_NAME} \
    --project=${PROJECT_ID} \
    --zone=${ZONE} \
    --machine-type=a2-highgpu-4g \
    --accelerator=count=4,type=nvidia-tesla-a100 \
    --image-family=ubuntu-accelerator-2204-amd64-with-nvidia-580 \
    --image-project=ubuntu-os-accelerator-images \
    --disk=name=${DISK_NAME},mode=rw,boot=no \
    --maintenance-policy=TERMINATE \
    --scopes=cloud-platform || true

echo "======================================================================="
echo "GPU VM Provisioned & Disk Attached!"
echo "Remote Execution Steps inside ${GPU_VM_NAME}:"
echo "1. Mount /dev/sdb to /mnt/disks/${DISK_NAME}"
echo "2. Run python3 scripts/verify_bf16_download.py --model-dir /mnt/disks/${DISK_NAME}/base_model"
echo "3. Run python3 scripts/run_cloud_reap_pipeline.py --model-dir /mnt/disks/${DISK_NAME}/base_model"
echo "4. Run python3 scripts/upload_hf.py --winner-dir artifacts/reap_run/winner_weights --repo-name True2456/Step-3.7-Flash-REAP-p15"
echo "5. Destroy VM: gcloud compute instances delete ${GPU_VM_NAME} --project=${PROJECT_ID} --zone=${ZONE} --quiet"
echo "======================================================================="
