# Foundation Models for digital pathology at your fingertips

In most computer vision fields, the challenge lies in algorithm development, and accessing images is straightforward. But in computational pathology, data scientists face unique hurdles. Simple tasks like storing, manipulating and loading images can be a time sink and source of frustration.

The challenges in computational pathology are many. Scanner vendors use proprietary file formats. Loading whole slide image (WSI) files from multiple vendors requires various code packages that are not well maintained. WSI files are storage intensive, holding gigabytes of data per file. Processing high magnification WSI files for downstream deep learning workflows requires cropping WSI files into many smaller images, turning a single WSI file into sometimes thousands of individual data products to track and maintain. Only after overcoming these hurdles and more can a data scientist start to build a model for the important task at hand– whether that’s detecting a specific biomarker, identifying mitoses, classifying a tumor type or any other of countless uses for AI in pathology.

However, with Proscia’s innovative Concentriq® Embeddings, these challenges are becoming relics of the past. AI development has recently pivoted from developing one-off models specifically tailored to specific tasks towards developing universal feature extractors, known as "foundation models", that learn representations from vast amounts of unlabeled data. With the success of language foundation models that power applications like ChatGPT, computer vision has followed suit with models like DINO and ConvNext, recognizing the immense potential of foundation models for accelerating downstream task-specific development. Now, the state of the art approach to many computer vision tasks in medical image analysis is to start by computing embeddings from images using a foundation model. Concentriq® Embeddings is now the first step in any computational pathology endeavor, leveraging vision foundation models to extract vital visual features from histopathology scans, turning cumbersome image files into standardized, lightweight feature vectors known as embeddings. Concentriq® Embeddings not only simplifies the process but also accelerates the development workflow in digital histopathology.

With Concentriq® Embeddings, Proscia is putting the power of foundation model embedding at developers’ fingertips to accelerate the digital histopathology image analysis workflow.

## Transforming Pathology with Concentriq® Embeddings

Proscia is proud to announce Concentriq® Embeddings, a seamless extension of our Concentriq® for Research platform. This tool is designed specifically for pharmaceutical companies, biotech companies, CROs, and academic research organizations to foster image-based research without the traditional barriers.
Concentriq® Embeddings is a backend service that provides foundation model embeddings from any slide in Concentriq® for Research. The service extracts rich visual features at any magnification, promptly providing access to the visual information in the slide, at a greatly compressed memory footprint.

Through Concentriq® Embeddings, developers can access some of the most widely used foundation models, and Proscia plans to continue adding to the list of supported foundation models. Instead of wading through the ever-growing and dense literature attempting to crown a “best” foundation model for pathology, Concentriq® Embeddings allows researchers to easily try out many feature extractors on a downstream task, and future-proofs for the inevitably better foundation models of tomorrow.

Currently supported foundation models include:

- DinoV2
  - Model Tag: `facebook/dinov2-base`
  - Patch Size: 224
  - Embedding Dimension: 768
  - License: [Apache-2.0](https://choosealicense.com/licenses/apache-2.0/)
  - [🤗 HuggingFace page](https://huggingface.co/facebook/dinov2-base)
  - [Paper](https://arxiv.org/abs/2304.07193)
- PLIP
  - Model Tag: `vinid/plip`
  - Patch Size: 224
  - Embedding Dimension: 512
  - License: [MIT](https://choosealicense.com/licenses/mit/)
  - [🤗 HuggingFace page](https://huggingface.co/vinid/plip)
  - [Paper](https://www.nature.com/articles/s41591-023-02504-3)
- ConvNext
  - Model Tag: `facebook/convnext-base-384-22k-1k`
  - Patch Size: 384
  - Embedding Dimension: 1024
  - License: [Apache-2.0](https://choosealicense.com/licenses/apache-2.0/)
  - [🤗 HuggingFace page](https://huggingface.co/facebook/convnext-base-384-22k-1k)
  - [Paper](https://arxiv.org/abs/2201.03545)
- CTransPath
  - Model Tag: `1aurent/swin_tiny_patch4_window7_224.CTransPath`
  - Patch Size: 224
  - Embedding Dimension: 768
  - License: [GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)
  - [🤗 HuggingFace page](https://huggingface.co/1aurent/swin_tiny_patch4_window7_224.CTransPath)
  - [Paper](https://www.sciencedirect.com/science/article/pii/S1361841522002043)
- H-optimus-0
  - Model Tag: `bioptimus/H-optimus-0`
  - Patch Size: 224
  - Embedding Dimension: 1536
  - License: [Apache-2.0](https://choosealicense.com/licenses/apache-2.0/)
  - [🤗 HuggingFace page](https://huggingface.co/bioptimus/H-optimus-0)
  - [Paper](https://github.com/bioptimus/releases/tree/main/models/h-optimus/v0)
- Virchow
  - Model Tag: `paige-ai/Virchow`
  - Patch Size: 224
  - Embedding Dimension: 2560
  - License: [Apache-2.0](https://choosealicense.com/licenses/apache-2.0/)
  - [🤗 HuggingFace page](https://huggingface.co/paige-ai/Virchow)
  - [Paper](https://arxiv.org/abs/2309.07778)

## Computational pathology development finally has a straightforward workflow

Concentriq® Embeddings revolutionizes the AI development process for everything from selecting WSIs to extracting features from them, making algorithm development more straightforward than ever.

_Before:_

> Data scientists juggled countless non-standardized WSI file formats and struggled with often poorly-maintained code packages for accessing whole slide image (WSI) files.

_Now with Concentriq® Embeddings:_

Forget about OpenSlide, proprietary SDKs from scanner vendors, and OpenPhi. Concentriq® for Research and Concentriq® Embeddings are all you need for your AI development.

---

_Before:_

> Training models for pathology images required downloading large WSI files and extensive storage capacity since each file often exceeds several gigabytes. This often produced more data than could be accommodated by a standard laptop hard drive. Furthermore, downloading such substantial amounts of data on a typical internet connection could take several hours or even days, significantly slowing down the research workflow and delaying critical advancements without costly specialty infrastructure.

_Now with Concentriq® Embeddings:_

Rather than managing slides that consume gigabytes of memory, data scientists and researchers now interact with lightweight feature representations that occupy just a few megabytes. For a concrete example: an RGB WSI crop of 512 pixels on each side contains 512x512x3= 786,432 unsigned 8-bit integers, or 786,432 bytes. In contrast, a Vision Transformer (ViT) feature vector (embedding) of this crop contains 768 floats at 4 bytes apiece, for 3,072 bytes. The feature vector is a compressed representation of the image, with a compression rate of 256! **This means a 1 Gb WSI becomes less than 4 Mb.**

---

_Before:_

> Preparing high magnification WSI files for downstream deep learning workflows involved cropping WSI files into many smaller images, turning a single WSI file into sometimes thousands of individual data products to track and maintain.

_Now with Concentriq® Embeddings:_

Data bookkeeping is greatly simplified. Concentriq® Embeddings tiles each slide and returns a single safetensor file per slide containing the embeddings. Even though the slide’s visual information is contained in a single convenient file, Concentriq® Embeddings provides an interface for loading feature vectors from individual crops into memory.

## This is how simple model development can be.

Discover the efficiency of the Concentriq® Embeddings workflow.

```python
from utils.client import ClientWrapper as Client

ce_api_client = Client(url=endpoint, email=email, password=pwd)
ticket_id = ce_api_client.embed_repos(ids=[1234], model="bioptimus/H-optimus-0", mpp=1)
embeddings = ce_api_client.get_embeddings(ticket_id)
```

# Setup

## Poetry & Pyenv

See instructions for using [pyenv](https://github.com/pyenv/pyenv-installer) for python environment management and installing [poetry](https://python-poetry.org/docs/) for dependency management and packaging in Python .

## Quickstart

```bash
pyenv local 3.10
poetry config virtualenvs.in-project true
poetry env use 3.10
poetry install --with dev
poetry shell
```
