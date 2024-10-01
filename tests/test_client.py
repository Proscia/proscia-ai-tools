import os
from unittest.mock import patch

import pytest
from requests.models import Response

from concentriq_embeddings_client.client import ConcentriqEmbeddingsClient
from concentriq_embeddings_client.models import (
    EstimationResponse,
    JobOutput,
    StatusResponse,
    SubmissionResponse,
)

BASE_URL = "https://concentriq-for-research.com"
TOKEN = os.getenv("CONCENTRIQ_TOKEN", default="fake_token")


def mock_response(status_code=200, json_data=None):
    """Helper function to create a mock Response object"""
    mock_resp = Response()
    mock_resp.status_code = status_code
    mock_resp._content = json_data.encode("utf-8")
    return mock_resp


@pytest.fixture
def client():
    return ConcentriqEmbeddingsClient(base_url=BASE_URL, token=TOKEN)


@patch("requests.Session.post")
def test_submit_job(mock_post, client):
    # Mock the API response
    mock_post.return_value = mock_response(
        json_data='{"ticket_id": "1234", "job_cost": 10.0, "credits_before_job": 100.0, "credits_after_job": 90.0}'
    )

    data = {"input_type": "image_ids", "input": [1, 2, 3], "model": "facebook/dinov2-base", "mpp": 1.0}

    response = client.submit_job(data)
    assert isinstance(response, SubmissionResponse)
    assert response.ticket_id == "1234"
    assert response.job_cost == 10.0


@patch("requests.Session.post")
def test_roi_selection(mock_post, client):
    mock_post.return_value = mock_response(json_data='{"ticket_id": "5678", "job_cost": 5.0}')

    data = {
        "image_id": 1,
        "regions": [{"height": 512, "width": 512, "x": 0, "y": 0}, {"height": 512, "width": 512, "x": 512, "y": 0}],
        "mpp": 0.5,
        "model": "facebook/dinov2-base",
    }

    response = client.roi_selection(data)
    assert isinstance(response, SubmissionResponse)
    assert response.ticket_id == "5678"
    assert response.job_cost == 5.0


@patch("requests.Session.post")
def test_estimate_job_cost(mock_post, client):
    mock_post.return_value = mock_response(
        json_data='{"job_cost": 10.0, "credits_before_job": 100.0, "credits_after_job": 90.0}'
    )

    data = {"input_type": "image_ids", "input": [1, 2, 3], "model": "facebook/dinov2-base", "mpp": 1.0}

    response = client.estimate_job_cost(data)
    assert isinstance(response, EstimationResponse)
    assert response.job_cost == 10.0
    assert response.credits_before_job == 100.0


@patch("requests.Session.get")
def test_get_job_status(mock_get, client):
    # Provide the correct mock data structure expected by StatusResponse
    mock_get.return_value = mock_response(
        json_data='{"status": "completed", "progress": 1.0, "finished": 10, "failed": 0, "queued": 0, "processing": 0}'
    )

    ticket = "1234"
    response = client.get_job_status(ticket)
    assert isinstance(response, StatusResponse)
    assert response.status == "completed"
    assert response.progress == 1.0
    assert response.finished == 10
    assert response.failed == 0
    assert response.queued == 0
    assert response.processing == 0


@patch("requests.Session.get")
def test_fetch_results(mock_get, client):
    mock_get.return_value = mock_response(json_data='{"images": [{"image_id": 1, "status": "completed"}]}')

    ticket = "1234"
    response = client.fetch_results(ticket)
    assert isinstance(response, JobOutput)
    assert len(response.images) == 1
    assert response.images[0].image_id == 1
    assert response.images[0].status == "completed"


@patch.object(ConcentriqEmbeddingsClient, "get_job_status")
@patch.object(ConcentriqEmbeddingsClient, "fetch_results")
def test_poll_for_completion_and_fetch_results(mock_fetch_results, mock_get_job_status):
    # Initialize the client with dummy data
    client = ConcentriqEmbeddingsClient(base_url="https://fakeurl.com", token=TOKEN)
    ticket = "1234"

    # Mock `get_job_status` to first return an in-progress job, then a completed job
    mock_get_job_status.side_effect = [
        StatusResponse(status="processing", progress=0.5),
        StatusResponse(status="completed", progress=1.0),
    ]

    # Mock `fetch_results` to return a JobOutput object
    mock_fetch_results.return_value = JobOutput(
        images=[{"image_id": 1, "status": "completed"}, {"image_id": 2, "status": "completed"}]
    )

    # Run the method under test
    result = client.poll_for_completion_and_fetch_results(ticket)

    # Assertions
    assert isinstance(result, JobOutput)
    assert len(result.images) == 2
    assert result.images[0].image_id == 1
    assert result.images[1].image_id == 2
    assert result.images[0].status == "completed"
    assert result.images[1].status == "completed"
