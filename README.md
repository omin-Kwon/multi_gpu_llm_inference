# LLM Inference Optimization on Multiple Nodes and GPUs

---

## Project Description

This project, **LLM Inference Optimization on Multiple Nodes and GPUs**, is the final project for the High Performance and Scalable Computing Spring class at Seoul National University (SNU). The objective is to perform efficient and scalable inference on a GPT-2 model using 16 GPUs across 4 nodes. This project leverages CUDA and MPI to achieve high performance.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Example](#example)
6. [Contributing](#contributing)
7. [License](#license)

---

## Introduction

This project demonstrates the implementation of a distributed GPT-2 inference engine. By utilizing 4 nodes with 4 GPUs each, we aim to optimize the performance and scalability of large language model (LLM) inference tasks. This implementation involves writing CUDA kernels and integrating MPI for inter-node communication. With the current code, you can achieve a throughput of 20,000 tokens per second.

---

## Prerequisites

Before you begin, ensure you have the following libraries and tools installed:

- MPI Library (e.g., OpenMPI)
- CUDA Toolkit

Ensure that your environment is properly configured with the necessary drivers and libraries for CUDA and MPI.

---

## Installation

### Setting up MPI

To install OpenMPI on your system, use the following commands:

For Ubuntu:

```bash
sudo apt update
sudo apt install openmpi-bin openmpi-common libopenmpi-dev
```

For CentOS:

```bash
sudo yum install openmpi openmpi-devel
```

### Setting up CUDA

Follow the instructions on the NVIDIA CUDA Toolkit website to download and install the appropriate version for your system.

---

## Usage

### Cloning the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/multi_gpu_llm_inference.git
cd multi_gpu_llm_inference
```

### Running the Inference

To run the GPT-2 inference on multiple nodes, use the following command:

```bash
mpirun -np 16 --hostfile hostfile ./run_inference.sh
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.
