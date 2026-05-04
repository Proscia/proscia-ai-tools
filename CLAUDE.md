# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python SDK for interacting with Proscia's Concentriq platform for digital pathology AI development. The library provides:

- **Concentriq Embeddings API**: Extract foundation model embeddings from whole slide images (WSIs)
- **Concentriq LS API**: Interact with the Concentriq LS platform for image/repository management, overlays, and annotations

## Build and Test Commands

```bash
# Install dependencies
poetry install

# Run all tests
pytest

# Run a single test file
pytest tests/test_client.py

# Run a specific test
pytest tests/test_client.py::test_submit_job

# Run with coverage
pytest --cov=proscia_ai_tools

# Type checking
mypy proscia_ai_tools

# Linting (auto-fixes enabled)
ruff check proscia_ai_tools
ruff format proscia_ai_tools
```

## Architecture

### Client Layer (`proscia_ai_tools/`)

- **`client.py`** - `ClientWrapper`: High-level wrapper for embeddings workflows. Handles authentication token refresh, job submission, polling, result caching, and embedding/thumbnail download/loading.

- **`concentriqlsclient.py`** - `ConcentriqLSClient`: Client for the Concentriq LS REST API. Manages images, repositories, overlays (heatmaps), annotations, and annotation classes. Uses paginated queries for large result sets.

- **`concentriq_embeddings_client/client.py`** - `ConcentriqEmbeddingsClient`: Low-level client for the embeddings service API endpoints (submit job, get status, fetch results, ROI selection).

### Supporting Modules

- **`annotations.py`**: Data models for Concentriq annotations (`ConcentriqAnnotation`, `AnnotationShape`, `AnnotationBounds`) and conversion utilities between pixel/viewport coordinates, mask-to-contour conversion, and Aperio XML export.

- **`utils.py`**: Image processing utilities - overlay creation, mask overlay, thumbnail tiling, embedding parsing, and evaluation metrics (IoU, Dice, confusion matrix).

### Authentication Pattern

Both clients use bearer token auth obtained via basic auth to `/api/v3/auth/token`. The `catch_auth_exceptions` decorator automatically refreshes tokens on HTTPError.

### Embeddings Workflow

1. Submit job with image/repository IDs, model tag, and MPP (microns per pixel)
2. Poll job status until complete
3. Fetch results (paginated) containing presigned S3 URLs
4. Download `.safetensors` files to local cache
5. Load tensors with `safe_open()` to PyTorch device

### Supported Foundation Models

DinoV2, PLIP, ConvNext, CTransPath, H-optimus-0, Virchow (see README for model tags and embedding dimensions).

## Code Style

- Line length: 120 characters
- Uses ruff for linting with flake8-compatible rules
- Type hints encouraged (mypy configured)
- Tests use pytest with mocking via `unittest.mock`
