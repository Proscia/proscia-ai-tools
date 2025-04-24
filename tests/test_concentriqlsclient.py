import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from proscia_ai_tools.concentriqlsclient import ConcentriqLSClient

BASE_URL = "https://concentriq-for-research.com"
EMAIL = "test@example.com"
PASSWORD = "password"  # pragma: allowlist secret # noqa: S105


def mock_response(status_code=200, json_data=None):
    """Helper function to create a mock Response object"""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json.loads(json_data)
    return mock_resp


@pytest.fixture
def ls_client():
    with patch("requests.request") as mock_post:
        mock_post.return_value = mock_response(json_data='{"token": "fake_token"}')
        yield ConcentriqLSClient(url=BASE_URL, email=EMAIL, password=PASSWORD)


@patch("requests.Session.post")
def test_create_overlay(mock_post, ls_client):
    mock_post.return_value = mock_response(json_data='{"data": {"id": 1, "name": "overlay"}}')
    response = ls_client.create_overlay(image_id=1, overlay_name="overlay")
    assert response["id"] == 1
    assert response["name"] == "overlay"


@patch("requests.Session.get")
def test_sign_overlay_s3_url(mock_get, ls_client):
    mock_get.return_value = mock_response(json_data='{"signed_request": "https://signed.url"}')
    signed_url = ls_client.sign_overlay_s3_url(overlay_id=1)
    assert signed_url == "https://signed.url"


@patch("proscia_ai_tools.utils.create_overlay")
@patch("proscia_ai_tools.utils.image_as_bytes")
@patch("requests.Session.post")
@patch("requests.Session.get")
@patch("requests.Session.put")
def test_insert_heatmap_overlay(mock_put, mock_get, mock_post, mock_image_as_bytes, mock_create_overlay, ls_client):
    mock_create_overlay.return_value = "fake_image"
    mock_image_as_bytes.return_value = b"fake_image_data"
    mock_put.return_value = mock_response(json_data="{}")
    mock_post.return_value = mock_response(json_data='{"data": {"id": 1}}')
    mock_get.return_value = mock_response(json_data='{"signed_request": "https://signed.url"}')
    ls_client.insert_heatmap_overlay(image_id=1, heatmap=np.array([[0, 1], [1, 0]]), result_name="heatmap")
    assert mock_post.called
    assert mock_get.called
    assert mock_put.called


@patch("requests.Session.get")
def test_get_image_data(mock_get, ls_client):
    mock_get.return_value = mock_response(json_data='{"data": {"id": 1, "name": "image"}}')
    image_data = ls_client.get_image_data(image_id=1)
    assert image_data["id"] == 1
    assert image_data["name"] == "image"


@patch("requests.Session.get")
def test_get_repo_data(mock_get, ls_client):
    mock_get.return_value = mock_response(json_data='{"data": {"id": 1, "name": "repo"}}')
    repo_data = ls_client.get_repo_data(target_repo_id="1")
    assert repo_data["id"] == 1
    assert repo_data["name"] == "repo"


@patch("requests.Session.get")
def test_paginated_get_query(mock_get, ls_client):
    mock_get.return_value = mock_response(json_data='{"data": {"items": [{"id": 1, "name": "item"}]}}')
    response = ls_client.paginated_get_query(url="https://fakeurl.com", params={})
    assert response["data"]["items"][0]["id"] == 1
    assert response["data"]["items"][0]["name"] == "item"


@patch("requests.Session.get")
def test_get_images_in_a_repo(mock_get, ls_client):
    mock_get.return_value = mock_response(json_data='{"data": {"images": [{"id": 1, "name": "image"}]}}')
    images = ls_client.get_images_in_a_repo(target_repo_id="1")
    assert images["images"][0]["id"] == 1
    assert images["images"][0]["name"] == "image"


@patch("requests.Session.post")
def test_create_annotation_class(mock_post, ls_client):
    mock_post.return_value = mock_response(json_data='{"id": 1, "name": "class"}')
    response = ls_client.create_annotation_class(annotation_class_name="class")
    assert response["id"] == 1
    assert response["name"] == "class"


@patch("requests.Session.post")
def test_add_annotation_class_to_repo(mock_post, ls_client):
    mock_post.return_value = mock_response(json_data='{"id": 1}')
    response = ls_client.add_annotation_class_to_repo(annotation_class_id=1, repo_id=1)
    assert response["id"] == 1


@patch("requests.Session.patch")
def test_assign_annotation_to_class(mock_patch, ls_client):
    mock_patch.return_value = mock_response(json_data='{"id": 1}')
    response = ls_client.assign_annotation_to_class(annotation_id=1, annotation_class_id=1)
    assert response["id"] == 1


@patch("requests.Session.post")
@patch("requests.Session.get")
def test_insert_annotations_from_mask(mock_get, mock_post, ls_client):
    mock_get.return_value = mock_response(json_data='{"data": {"mppx": 0.5, "imgWidth": 1000, "imgHeight": 1000}}')
    mock_post.return_value = mock_response(json_data='{"id": 1}')
    mask = np.zeros((100, 100))
    mask[10:20, 10:20] = 1
    annotations = ls_client.insert_annotations_from_mask(image_id=1, mask=mask, mask_mpp=0.5)
    assert annotations[0]["id"] == 1


@patch("requests.Session.get")
def test_get_annotations(mock_get, ls_client):
    mock_get.return_value = mock_response(json_data='{"data": {"annotations": [{"id": 1, "name": "annotation"}]}}')
    annotations = ls_client.get_annotations(image_ids=[1])
    assert annotations["annotations"][0]["id"] == 1
    assert annotations["annotations"][0]["name"] == "annotation"
