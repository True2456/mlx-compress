#!/usr/bin/env bash
# ==============================================================================
# GCP Phase 1: CPU Disk Staging & BF16 Base Model Download ($0.03 Total Cost)
# ==============================================================================
# Project: project-e60cdf5e-8d01-4a57-9f0
# Region: us-central1
# VM: e2-standard-4 ($0.13/hr)
# Disk: 1.5 TB SSD (step37-disk)
# ==============================================================================

set -euo pipefail

GCLOUD_BIN="$(which gcloud 2>/dev/null || echo "/Users/true/google-cloud-sdk/bin/gcloud")"
PROJECT_ID="project-e60cdf5e-8d01-4a57-9f0"
ZONE="us-central1-a"
VM_NAME="step37-cpu-staging"
DISK_NAME="step37-disk"
DISK_SIZE="1500GB"

echo "======================================================================="
echo "PHASE 1: Provisioning 1.5 TB Persistent Disk & Cheap CPU Staging VM"
echo "======================================================================="

# 1. Create Persistent Disk (1.5 TB Standard)
echo "📦 Creating 1.5 TB Persistent Disk: ${DISK_NAME}..."
${GCLOUD_BIN} compute disks create ${DISK_NAME} \
    --project=${PROJECT_ID} \
    --zone=${ZONE} \
    --type=pd-standard \
    --size=${DISK_SIZE} || true

# 2. Create CPU Staging VM
echo "🖥️ Provisioning cheap CPU VM (${VM_NAME})..."
${GCLOUD_BIN} compute instances create ${VM_NAME} \
    --project=${PROJECT_ID} \
    --zone=${ZONE} \
    --machine-type=e2-standard-4 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --disk=name=${DISK_NAME},mode=rw,boot=no \
    --scopes=cloud-platform

echo "======================================================================="
echo "CPU VM Provisioned. Run remote download commands inside VM:"
echo "1. Format & mount /dev/sdb to /mnt/disks/${DISK_NAME}"
echo "2. huggingface-cli download stepfun-ai/Step-3.7-Flash --local-dir /mnt/disks/${DISK_NAME}/base_model"
echo "3. python3 scripts/verify_bf16_download.py --model-dir /mnt/disks/${DISK_NAME}/base_model"
echo "======================================================================="
