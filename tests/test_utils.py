import imageio.v2 as imageio
import numpy as np
import pytest

from proscia_ai_tools.utils import (
    ShapeMismatchException,
    SingleChannelException,
    calculate_boolean_metrics,
    calculate_dice,
    calculate_iou,
    parse,
    tile_thumbnail,
)


def test_overlay_mask():
    import numpy as np

    from proscia_ai_tools.utils import overlay_mask

    blend = np.random.rand(100, 100, 3)
    mask = np.random.rand(100, 100)
    overlay = overlay_mask(blend, mask)
    assert overlay.shape == (100, 100, 3)


def test_overlay_mask_wrong_shape():
    import numpy as np

    from proscia_ai_tools.utils import overlay_mask

    blend = np.random.rand(100, 100, 3)
    mask = np.random.rand(150, 150, 3)
    with pytest.raises(ShapeMismatchException):
        overlay_mask(blend, mask)


def test_overlay_mask_wrong_channel():
    import numpy as np

    from proscia_ai_tools.utils import overlay_mask

    blend = np.random.rand(100, 100, 3)
    mask = np.random.rand(100, 100, 3)
    with pytest.raises(SingleChannelException):
        overlay_mask(blend, mask)


def test_parse():
    import torch

    from proscia_ai_tools.utils import parse

    emb_dict = {"0_0": torch.rand(10), "0_1": torch.rand(10)}
    sorted_keys, coordinates = parse(emb_dict)
    assert sorted_keys == ["0_0", "0_1"]
    assert coordinates == {"0_0": [0, 0], "0_1": [0, 1]}


def test_stack_embedding():
    import torch

    from proscia_ai_tools.utils import stack_embedding

    emb_dict = {"0_0": torch.rand(10), "0_1": torch.rand(10)}
    sorted_keys, coordinates = parse(emb_dict)
    mat = stack_embedding(emb_dict, sorted_keys)
    assert mat.shape == (2, 10)


@pytest.fixture
def mock_embedding():
    return {
        "patch_size": 256,
        "thumb_mpp": 7.0,
        "mpp": 7.0,
        "local_thumbnail_path": "path/to/mock_thumbnail.png",
        "embedding": {
            "0_0": np.random.rand(10),
            "0_1": np.random.rand(10),
            "1_0": np.random.rand(10),
            "1_1": np.random.rand(10),
        },
    }


@pytest.fixture
def mock_thumbnail():
    # Create a mock thumbnail image of size 512x512 with 3 color channels
    return np.random.rand(512, 512, 3).astype(np.uint8)


def test_tile_thumbnail_basic(mock_embedding, mock_thumbnail, monkeypatch):
    def mock_imageio_imread(path):
        return mock_thumbnail

    monkeypatch.setattr(imageio, "imread", mock_imageio_imread)

    emb = mock_embedding
    tiles = tile_thumbnail(emb)
    assert len(tiles) == 4
    assert all(tile.shape == (256, 256, 3) for tile in tiles.values())


def test_tile_thumbnail_indices(mock_embedding, mock_thumbnail, monkeypatch):
    def mock_imageio_imread(path):
        return mock_thumbnail

    monkeypatch.setattr(imageio, "imread", mock_imageio_imread)

    emb = mock_embedding
    indices = [(0, 0), (1, 1)]
    tiles = tile_thumbnail(emb, indices)
    assert len(tiles) == 2
    assert all(tile.shape == (256, 256, 3) for tile in tiles.values())


def test_calculate_boolean_metrics_basic():
    gt = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    pred = np.array([1, 0, 1, 0, 0, 1, 1, 0])
    metrics = calculate_boolean_metrics(gt, pred)

    assert metrics["tp"] == 3
    assert metrics["fp"] == 1
    assert metrics["tn"] == 3
    assert metrics["fn"] == 1
    assert np.isclose(metrics["sen"], 0.75, atol=1e-5)
    assert np.isclose(metrics["spe"], 0.75, atol=1e-5)
    assert np.isclose(metrics["ppv"], 0.75, atol=1e-5)
    assert np.isclose(metrics["npv"], 0.75, atol=1e-5)
    assert np.isclose(metrics["acc"], 0.75, atol=1e-2)
    assert np.isclose(metrics["f1"], 0.75, atol=1e-2)


def test_calculate_boolean_metrics_edge_case_all_zeros():
    gt = np.array([0, 0, 0, 0])
    pred = np.array([0, 0, 0, 0])
    metrics = calculate_boolean_metrics(gt, pred)

    assert metrics["tp"] == 0
    assert metrics["fp"] == 0
    assert metrics["tn"] == 4
    assert metrics["fn"] == 0
    assert np.isclose(metrics["sen"], 0.0, atol=1e-5)
    assert np.isclose(metrics["spe"], 1.0, atol=1e-5)
    assert np.isclose(metrics["ppv"], 0.0, atol=1e-5)
    assert np.isclose(metrics["npv"], 1.0, atol=1e-5)
    assert np.isclose(metrics["acc"], 1.0, atol=1e-2)
    assert np.isclose(metrics["f1"], 0.0, atol=1e-2)


def test_calculate_boolean_metrics_edge_case_all_ones():
    gt = np.array([1, 1, 1, 1])
    pred = np.array([1, 1, 1, 1])
    metrics = calculate_boolean_metrics(gt, pred)

    assert metrics["tp"] == 4
    assert metrics["fp"] == 0
    assert metrics["tn"] == 0
    assert metrics["fn"] == 0
    assert np.isclose(metrics["sen"], 1.0, atol=1e-5)
    assert np.isclose(metrics["spe"], 0.0, atol=1e-5)
    assert np.isclose(metrics["ppv"], 1.0, atol=1e-5)
    assert np.isclose(metrics["npv"], 0.0, atol=1e-5)
    assert np.isclose(metrics["acc"], 1.0, atol=1e-2)
    assert np.isclose(metrics["f1"], 1.0, atol=1e-2)


def test_calculate_boolean_metrics_non_numpy_input():
    gt = [1, 0, 1, 0, 1, 0, 1, 0]
    pred = [1, 0, 1, 0, 0, 1, 1, 0]
    metrics = calculate_boolean_metrics(gt, pred)

    assert metrics["tp"] == 3
    assert metrics["fp"] == 1
    assert metrics["tn"] == 3
    assert metrics["fn"] == 1
    assert np.isclose(metrics["sen"], 0.75, atol=1e-5)
    assert np.isclose(metrics["spe"], 0.75, atol=1e-5)
    assert np.isclose(metrics["ppv"], 0.75, atol=1e-5)
    assert np.isclose(metrics["npv"], 0.75, atol=1e-5)
    assert np.isclose(metrics["acc"], 0.75, atol=1e-2)
    assert np.isclose(metrics["f1"], 0.75, atol=1e-2)


def test_calculate_iou_basic():
    gt = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    pred = np.array([1, 0, 1, 0, 0, 1, 1, 0])
    iou = calculate_iou(gt, pred)
    assert np.isclose(iou, 0.6, atol=1e-5)


def test_calculate_iou_all_zeros():
    gt = np.array([0, 0, 0, 0])
    pred = np.array([0, 0, 0, 0])
    iou = calculate_iou(gt, pred)
    assert np.isclose(iou, 1.0, atol=1e-5)  # IoU should be 1 when both are all zeros


def test_calculate_iou_all_ones():
    gt = np.array([1, 1, 1, 1])
    pred = np.array([1, 1, 1, 1])
    iou = calculate_iou(gt, pred)
    assert np.isclose(iou, 1.0, atol=1e-5)  # IoU should be 1 when both are all ones


def test_calculate_dice_basic():
    gt = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    pred = np.array([1, 0, 1, 0, 0, 1, 1, 0])
    dice = calculate_dice(gt, pred)
    assert np.isclose(dice, 0.75, atol=1e-5)


def test_calculate_dice_all_zeros():
    gt = np.array([0, 0, 0, 0])
    pred = np.array([0, 0, 0, 0])
    dice = calculate_dice(gt, pred)
    assert np.isclose(dice, 1.0, atol=1e-5)  # Dice should be 1 when both are all zeros


def test_calculate_dice_all_ones():
    gt = np.array([1, 1, 1, 1])
    pred = np.array([1, 1, 1, 1])
    dice = calculate_dice(gt, pred)
    assert np.isclose(dice, 1.0, atol=1e-5)  # Dice should be 1 when both are all ones
