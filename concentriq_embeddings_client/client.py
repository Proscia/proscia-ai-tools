import time
from typing import Dict

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from concentriq_embeddings_client.models import (
    EstimationResponse,
    JobOutput,
    StatusResponse,
    SubmissionResponse,
    ThumbnailsJobOutput,
)


class ConcentriqEmbeddingsClient:
    def __init__(self, base_url: str, token: str, api_version: str = "v1"):
        """Client for the Concentriq Embeddings service.

        Args:
            base_url (str): The base URL of the embeddings service
            token (str): The API token. Your token can be obtained via basic auth
                using the `<base_url>/api/v3/auth/token endpoint.
            api_version (str): The embeddings API version

        Example:
            client = ConcentriqEmbeddingsClient(
                base_url="https://concentriq-for-research.com",
                token="your_token_here",
                api_version="v1"
            )
        """
        self.base_url = base_url
        self.token = token
        self.api_version = api_version
        self.session = requests.Session()
        retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
        self.session.mount("http://", HTTPAdapter(max_retries=retries))
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        self.session.headers.update({"token": self.token})

    def submit_job(self, data: Dict, thumbnails: bool = False) -> SubmissionResponse:
        """Method to submit a job to the embeddings service.
        Optionally submit a job to get thumbnails.

        Args:
            data (Dict): The input data
            thumbnails (bool): Whether to get thumbnails or not (default is embeddings)
        Returns:
            SubmissionResponse: The response object

        Example (embeddings):
            data = {
                "input_type": "image_ids",
                "input": [1,2,3],
                "model": "facebook/dinov2-base",
                "mpp": 1.0
            }
            response = client.submit_job(data)
        """
        maybe_thumbnails = "/thumbnails" if thumbnails else ""
        url = f"{self.base_url}/embeddings/{self.api_version}{maybe_thumbnails}/submit-job/"
        response = self.session.post(url, json=data)
        response.raise_for_status()
        return SubmissionResponse(**response.json())

    def roi_selection(self, data: Dict) -> SubmissionResponse:
        """Request embeddings for specific regions of interest of a slide

        Args:
            data (Dict): The input data
        Returns:
            SubmissionResponse: The response object

        Example (embeddings):
        data = {
        "image_id": 1,
        "regions": [
            {
            "height": 512,
            "width": 512,
            "x": 0,
            "y": 0
            },
            {
            "height": 512,
            "width": 512,
            "x": 512,
            "y": 0
            }
        ],
        "mpp": 0.5,
        "model": "facebook/dinov2-base"
        }
            response = client.submit_job(data)
        """
        url = f"{self.base_url}/embeddings/{self.api_version}/roi-selection/"
        response = self.session.post(url, json=data)
        response.raise_for_status()
        return SubmissionResponse(**response.json())

    def estimate_job_cost(self, data: Dict) -> EstimationResponse:
        """Method to estimate a job cost.

        Args:
            data (Dict): The input data
        Returns:
            EstimationResponse: The response object

        Example:
            data = {
                "input_type": "image_ids",
                "input": [1,2,3],
                "model": "facebook/dinov2-base",
                "mpp": 1.0
            }
            response = client.estimate_job_duation(data)
        """
        url = f"{self.base_url}/embeddings/{self.api_version}/estimate-job/"
        response = self.session.post(url, json=data)
        response.raise_for_status()
        return EstimationResponse(**response.json())

    def get_job_status(self, ticket: str, thumbnails: bool = False) -> StatusResponse:
        """Method to get the status of a job.

        Args:
            ticket (str): The job ticket
            thumbnails (bool): Whether to get the status of a thumbnail job or not (default is embeddings)

        Returns:
            StatusResponse: The response object
        """
        maybe_thumbnails = "/thumbnails" if thumbnails else ""
        url = f"{self.base_url}/embeddings/{self.api_version}{maybe_thumbnails}/status/{ticket}"
        response = self.session.get(url)
        response.raise_for_status()
        return StatusResponse(**response.json())

    def fetch_results(self, ticket: str, offset: int = 0, limit: int = 100, thumbnails: bool = False) -> JobOutput:
        """Method to fetch results of a job.

        Args:
            ticket (str): The job ticket
            offset (int): The offset for pagination
            limit (int): The limit for pagination

        Returns:
            JobOutput: The response object

        """
        maybe_thumbnails = "/thumbnails" if thumbnails else ""
        url = f"{self.base_url}/embeddings/{self.api_version}{maybe_thumbnails}/results/{ticket}?offset={offset}&limit={limit}"
        response = self.session.get(url)
        response.raise_for_status()
        if thumbnails:
            return ThumbnailsJobOutput(**response.json())
        else:
            return JobOutput(**response.json())

    def poll_for_completion_and_fetch_results(self, ticket: str, check_interval: int = 5) -> JobOutput:
        """Polls job status and fetches results once complete.

        Args:
            ticket (str): The job ticket
            check_interval (int): The interval in seconds between status checks

        Returns:
            JobOutput: The response object
        """
        while True:
            status = self.get_job_status(ticket)
            if status.progress == 1.0:  # Check if job progress is 100%
                break
            time.sleep(check_interval)  # Wait before the next status check

        # Fetch all results
        all_results = []
        offset = 0
        limit = 1000
        while True:
            results = self.fetch_results(ticket, offset, limit)
            all_results.extend(results.images)
            if len(results.images) < limit:
                break
            offset += limit

        return JobOutput(images=all_results)
