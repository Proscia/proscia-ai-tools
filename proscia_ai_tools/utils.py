import io
from typing import Any, Dict, List, Tuple

import cv2
import imageio
import numpy as np
import torch
from PIL import Image


class ShapeMismatchException(Exception):
    def __init__(self, thumbnail_shape: Tuple, mask_shape: Tuple) -> None:
        super().__init__(f"Thumbnail shape {thumbnail_shape} != mask shape {mask_shape}.")


class SingleChannelException(Exception):
    def __init__(self) -> None:
        super().__init__("Mask must be single channel.")


def create_overlay(mat: np.ndarray) -> np.ndarray:
    """Create 4 channel image of `mat` as a heatmap.
    Sets the alpha channel equal to zero where `mat` is zero

    Parameters
    ----------
    mat : np.ndarray
        Matrix to convert to heatmap. Must be 2D floating point and range from 0-1

    Returns
    -------
    np.ndarray
        Heatmap image with alpha channel
    """
    heatmap = cv2.applyColorMap((mat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2RGBA)
    heatmap[..., 3] = (mat / mat.max() * 255).astype(np.uint8)
    heatmap[mat == 0, 0] = 255
    heatmap[mat == 0, 1] = 255
    heatmap[mat == 0, 2] = 255
    return heatmap


def image_as_bytes(overlay: np.ndarray) -> bytes:
    """Converts an image array to bytes.

    Parameters
    ----------
    overlay : np.ndarray
        Image array

    Returns
    -------
    bytes
        Image bytes
    """
    overlay = Image.fromarray(np.uint8(overlay))
    overlay_bytes = io.BytesIO()
    overlay.save(overlay_bytes, format="PNG", optimize=True, quality=85)
    overlay_bytes = overlay_bytes.getvalue()
    return overlay_bytes


def overlay_mask(
    thumbnail: np.ndarray, mask: np.ndarray, rgb_color: Tuple[int, int, int] = (0, 240, 0), alpha: float = 0.4
) -> np.ndarray:
    """Overlays a boolean mask onto a thumbnail.

    Parameters
    ----------
    thumbnail : np.ndarray
        Thumbnail
    mask : np.ndarray
        Mask
    rgb_color : Tuple[int,int,int], optional
        Color of mask as RGB triplet, by default (0,240,0)
    alpha : float, optional
        Alpha to blend thumbnail and mask, by default 0.4

    Returns
    -------
    np.ndarray
        Thumbnail with overlaid mask

    Raises
    ------
    ShapeMismatchException
        Thumbnail and mask have different x-y dimensions
    SingleChannelException
        Mask contains more than one channel
    """
    if thumbnail.shape[:2] != mask.shape[:2]:
        raise ShapeMismatchException(thumbnail.shape[:2], mask.shape[:2])
    mask = np.squeeze(mask).astype(bool)
    if len(mask.shape) > 2:
        raise SingleChannelException()
    rgb_color = np.reshape(np.array(rgb_color), (1, 1, 3)).astype(np.uint8)
    blend = np.ubyte((1 - alpha) * thumbnail + alpha * rgb_color)
    mask = np.repeat(np.expand_dims(mask, axis=2), 3, axis=2)
    overlay = (blend * mask) + (thumbnail * ~mask)
    return overlay


def parse(emb_dict: Dict[str, torch.Tensor]) -> Tuple[List[str], Dict[str, List[int]]]:
    """
    Parse the embeddings dictionary into sorted keys and a dictionary of coordinates.

    Parameters
    ----------
    emb_dict (Dict[str, torch.Tensor]): The embeddings dictionary.

    Returns
    -------
    Tuple[List[str], Dict[str, List[int]]]: A tuple of a list of sorted keys and a dictionary of coordinates.
    """
    sorted_keys = sorted(emb_dict.keys())
    coordinates = {k: [int(loc) for loc in k.split("_")] for k in sorted_keys}
    return sorted_keys, coordinates


def stack_embedding(embedding: Dict[str, Any], skeys: List[str]) -> np.ndarray:
    """
    Stack the embeddings into a matrix.

    Parameters
    ----------
    embedding (dict): The embeddings dictionary.
    skeys (List[str]): A list of sorted keys.

    Returns
    -------
    np.ndarray: A matrix of embeddings.
    """
    vector_mat = np.stack([embedding[k].cpu().numpy() for k in skeys])
    return vector_mat


def tile_thumbnail(emb: dict, indices=None) -> Dict[str, np.ndarray]:
    """
    Tile the thumbnail image into tiles.

    Parameters
    ----------
    emb (dict): The embeddings dictionary.
    indices (List[Tuple[int, int]], optional): A list of indices to tile. Defaults to None which tiles the entire thumbnail.

    Returns
    -------
    Dict[str, np.ndarray]: A dictionary of tiles.
    """
    patch_size = emb["patch_size"]
    thumb_mpp = emb["thumb_mpp"]
    emb_mpp = emb["mpp"]
    thumbnail = imageio.v2.imread(emb["local_thumbnail_path"])
    thumb_px_per_tile = int((patch_size * emb_mpp) / thumb_mpp)
    h, w = thumbnail.shape[:2]
    pad_h = thumb_px_per_tile - (h % thumb_px_per_tile)
    pad_w = thumb_px_per_tile - (w % thumb_px_per_tile)
    thumb = np.stack(
        [cv2.copyMakeBorder(thumbnail[:, :, i], 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=255) for i in range(3)],
        axis=2,
    )

    if indices is None:
        _, locdict = parse(emb["embedding"])
        indices = locdict.values()

    tiles = {}
    for i, j in indices:
        tile = thumb[
            (i * thumb_px_per_tile) : ((i + 1) * (thumb_px_per_tile)),
            (j * thumb_px_per_tile) : ((j + 1) * thumb_px_per_tile),
            :,
        ]
        tiles.update({f"{i}_{j}": tile})
    return tiles


def calculate_boolean_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    """Calculates boolean metrics on arrays of ground truth labels and predictions.

    Parameters
    ----------
    gt : np.ndarray
        Ground truth array, any shape
    pred : np.ndarray
        Prediction array, same shape as gt

    Returns
    -------
    dict
        Metrics dict, returning cm, sensitivity, specificity, ppv, npv
    """
    # input validation
    if not isinstance(gt, np.ndarray):
        gt = np.array(gt)
    if not isinstance(pred, np.ndarray):
        pred = np.array(pred)
    # use GT and pred to calculate metrics
    cm = np.bincount(np.ravel(gt) * 2 + np.ravel(pred), minlength=4).reshape(2, 2)
    metrics_dict = {}
    metrics_dict["cm"] = cm
    metrics_dict["tp"] = cm[1][1]
    metrics_dict["fp"] = cm[0][1]
    metrics_dict["tn"] = cm[0][0]
    metrics_dict["fn"] = cm[1][0]
    metrics_dict["sen"] = metrics_dict["tp"] / (metrics_dict["tp"] + metrics_dict["fn"] + 1e-5)
    metrics_dict["spe"] = metrics_dict["tn"] / (metrics_dict["tn"] + metrics_dict["fp"] + 1e-5)
    metrics_dict["ppv"] = metrics_dict["tp"] / (metrics_dict["tp"] + metrics_dict["fp"] + 1e-5)
    metrics_dict["npv"] = metrics_dict["tn"] / (metrics_dict["tn"] + metrics_dict["fn"] + 1e-5)
    metrics_dict["acc"] = np.round((metrics_dict["tp"] + metrics_dict["tn"]) / (np.sum(metrics_dict["cm"]) + 1e-5), 2)
    metrics_dict["f1"] = np.round(
        2 * ((metrics_dict["sen"] * metrics_dict["ppv"]) / (metrics_dict["sen"] + metrics_dict["ppv"] + 1e-5)), 2
    )
    return metrics_dict


def calculate_iou(gt: np.ndarray, pred: np.ndarray, eps=1e-9):
    """
    Calculate Intersection over Union

    Parameters
    ----------

    gt : np.ndarray
        Ground truth array, any shape
    pred : np.ndarray

    Returns
    -------
    float
        IoU score
    """
    intersection = np.sum((gt == 1) & (pred == 1))
    union = np.sum((gt == 1) | (pred == 1))
    return (intersection + eps) / (union + eps)


def calculate_dice(gt: np.ndarray, pred: np.ndarray, eps=1e-9):
    """
    Calculate Dice score

    Parameters
    ----------

    gt : np.ndarray
        Ground truth array, any shape
    pred : np.ndarray

    Returns
    -------
    float
        Dice score
    """
    intersection = 2 * np.sum((gt == 1) & (pred == 1))
    sums = np.sum(gt == 1) + np.sum(pred == 1)
    return (intersection + eps) / (sums + eps)
