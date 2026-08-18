import base64
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from proscia_ai_tools.concentriq_embeddings_client.models import (
    EstimationResponse,
    JobOutput,
    ModelsListResponse,
    StatusResponse,
    SubmissionResponse,
    ThumbnailsJobOutput,
)

API_KEY_HEADER_NAME = "concentriq-api-key"  # pragma: allowlist secret


class DetailedHTTPError(requests.exceptions.HTTPError):
    """Custom HTTP error class to provide detailed error messages."""

    def __init__(self, response: requests.Response):
        self.response = response
        self.status_code = response.status_code

        try:
            error_data = response.json()
            self.message = error_data.get("message") or error_data.get("error") or str(error_data)
        except ValueError:
            self.message = response.text.strip()

        super().__init__(f"{self.status_code} Error: {self.message}", response=response)


class ConcentriqEmbeddingsClient:
    def __init__(
        self,
        base_url: str,
        token: str | None = None,
        api_version: str = "v1",
        *,
        email: str | None = None,
        password: str | None = None,
        api_key: str | None = None,
    ):
        """Client for the Concentriq Embeddings service.

        Provide exactly one of the following auth methods:
          - ``token``: A JWT bearer token (from ``/api/v3/auth/token``).
          - ``email`` + ``password``: Basic auth credentials.
          - ``api_key``: A Concentriq API key.

        Args:
            base_url: The base URL of the embeddings service.
            token: JWT bearer token.
            api_version: The embeddings API version.
            email: Email for basic auth.
            password: Password for basic auth.
            api_key: Concentriq API key.
        """
        self.base_url = base_url
        self.api_version = api_version
        self.session = requests.Session()
        retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
        self.session.mount("http://", HTTPAdapter(max_retries=retries))
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

        has_basic = email is not None and password is not None
        has_token = token is not None
        has_api_key = api_key is not None
        provided = sum([has_basic, has_token, has_api_key])
        if provided != 1:
            raise ValueError("Provide exactly one auth method: token, email+password, or api_key.")  # noqa: TRY003

        if has_token:
            self.session.headers.update({"Authorization": f"Bearer {token}"})
        elif has_basic:
            encoded = base64.b64encode(f"{email}:{password}".encode()).decode()
            self.session.headers.update({"Authorization": f"Basic {encoded}"})
        elif has_api_key:
            self.session.headers.update({API_KEY_HEADER_NAME: api_key})

    def submit_job(self, data: dict, thumbnails: bool = False) -> SubmissionResponse:
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
        try:
            response = self.session.post(url, json=data)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            raise DetailedHTTPError(response) from e
        return SubmissionResponse(**response.json())

    def roi_selection(self, data: dict) -> SubmissionResponse:
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
        try:
            response = self.session.post(url, json=data)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            raise DetailedHTTPError(response) from e
        return SubmissionResponse(**response.json())

    def estimate_job_cost(self, data: dict) -> EstimationResponse:
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
        try:
            response = self.session.post(url, json=data)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            raise DetailedHTTPError(response) from e
        return EstimationResponse(**response.json())

    def list_models(self) -> ModelsListResponse:
        """Method to list the foundation models available for creating embeddings in this deployment.

        Returns:
            ModelsListResponse: The response object
        """
        url = f"{self.base_url}/embeddings/{self.api_version}/models/"
        try:
            response = self.session.get(url)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            raise DetailedHTTPError(response) from e
        return ModelsListResponse(**response.json())

    def get_job_status(self, ticket: str, thumbnails: bool = False) -> StatusResponse:
        """Method to get the status of a job.

        Args:
            ticket (str): The job ticket
            thumbnails (bool): Whether to get the status of a thumbnail job or not (default is embeddings)

        Returns:
            StatusResponse: The response object
        """
        maybe_thumbnails = "/thumbnails" if thumbnails else ""
        url = f"{self.base_url}/embeddings/{self.api_version}{maybe_thumbnails}/status/{ticket}/"
        try:
            response = self.session.get(url)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            raise DetailedHTTPError(response) from e
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
        url = f"{self.base_url}/embeddings/{self.api_version}{maybe_thumbnails}/results/{ticket}/?offset={offset}&limit={limit}"
        try:
            response = self.session.get(url)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            raise DetailedHTTPError(response) from e
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
