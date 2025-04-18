import json
import os
import time
from functools import wraps
from typing import Dict, List

import imageio
import numpy as np
import requests
from requests.auth import HTTPBasicAuth
from requests.models import HTTPError
from safetensors import safe_open

from concentriq_embeddings_client.client import ConcentriqEmbeddingsClient


class GetException(Exception):
    """Exception raised when get request fails."""

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class ClientWrapper:
    """
    A wrapper around the Concentriq Embeddings client.

    Parameters
    ----------
    url (str): The URL of the Concentriq Embeddings endpoint.
    email (str): The email address to use for authentication.
    password (str): The password to use for authentication.
    cache_dir (str, optional): The directory to use for caching results. Defaults to `./data`.
    """

    def __init__(self, url: str, email: str, password: str, cache_dir="./data", device: int = 0, api_version="v1"):
        self.cache_dir = cache_dir
        self.device = device
        self.base_url = url
        self.email = email
        self.password = password
        self.api_version = api_version
        self.refresh_client_token()

    def catch_auth_exceptions(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except HTTPError:
                self.refresh_client_token()
                return func(self, *args, **kwargs)

        return wrapper

    def refresh_client_token(self):
        self._get_token()
        self.client = ConcentriqEmbeddingsClient(base_url=self.base_url, token=self.token, api_version=self.api_version)

    def _get_token(self) -> None:
        url = f"{self.base_url}/api/v3/auth/token"
        payload = {}
        response = requests.request(
            "POST", url, auth=HTTPBasicAuth(username=self.email, password=self.password), data=payload
        )
        self.token = response.json().get("token", None)

    @catch_auth_exceptions
    def _submit_job(self, data: Dict, thumbnails: bool = False) -> str:
        """
        Submits a job to the Concentriq Embeddings (or thumbnails) endpoint.

        Parameters
        ----------
        data (dict): The data to submit.
        thumbnails (bool, optional): Whether to submit a job to request thumbnails instead of embeddings. Defaults to False.

        Returns
        -------
        str: The ticket ID of the submitted job."""
        submit_job_response = self.client.submit_job(data, thumbnails=thumbnails)
        return submit_job_response.ticket_id

    def embed_images(self, ids: List[int], mpp: float = 1, model: str = "facebook/dinov2-base") -> str:
        """
        Embeds a list of images.

        Parameters
        ----------
        ids (List[int]): The list of image IDs.
        mpp (float, optional): The microns per pixel. Defaults to 1.
        model (str, optional): The model to use. Defaults to "facebook/dinov2-base".

        Returns
        -------
        str: The ticket ID of the submitted job.
        """
        data = {"input_type": "image_ids", "input": ids, "mpp": mpp, "model": model}
        return self._submit_job(data)

    def thumbnail_images(self, ids: List[int]) -> str:
        """
        Submits a job to produce thumbnails of a list of images.

        Parameters
        ----------
        ids (List[int]): The list of image IDs.

        Returns
        -------
        str: The ticket ID of the submitted job.
        """
        data = {"input_type": "image_ids", "input": ids}
        return self._submit_job(data, thumbnails=True)

    def thumbnail_repos(self, ids: List[int]) -> str:
        """
        Submits a job to produce thumbnails of a list of repository ids.

        Parameters
        ----------
        ids (List[int]): The list of repository IDs.

        Returns
        -------
        str: The ticket ID of the submitted job.
        """
        data = {"input_type": "repository_ids", "input": ids}
        return self._submit_job(data, thumbnails=True)

    @catch_auth_exceptions
    def embed_roi(self, image_id: int, regions: List[Dict], mpp: float = 1, model: str = "facebook/dinov2-base") -> str:
        """
        Request embeddings for specific regions of interest of a slide.

        Parameters
        ----------
        image_id (int): The image_id of the requested slide.
        regions (List[Dict]): The regions of interest specified as a list of bounding boxes of the following format:
          ```[
                {
                "height": 512,
                "width": 512,
                "x": 0,
                "y": 0
                },
                ...
            ]```
            Where "x" and "y" are the top left corner of the bounding box and "height" and "width" are the dimensions of the bounding box.
            All values are defined at 7mpp resolution to be compatible with the thumbnail endpoint.
        mpp (float, optional): The microns per pixel. Defaults to 1.
        model (str, optional): The model to use. Defaults to "facebook/dinov2-base".

        Returns
        -------
        str: The ticket ID of the submitted job.
        """
        data = {"image_id": image_id, "regions": regions, "mpp": mpp, "model": model}
        return self.client.roi_selection(data).ticket_id

    def embed_repos(self, ids: List[int], mpp: float = 1, model: str = "facebook/dinov2-base") -> str:
        """
        Embeds a list of repositories.

        Parameters
        ----------
        ids (List[int]): The list of repository IDs.
        mpp (float, optional): The microns per pixel. Defaults to 1.
        model (str, optional): The model to use. Defaults to "facebook/dinov2-base".

        Returns
        -------
        str: The ticket ID of the submitted job.
        """
        data = {"input_type": "repository_ids", "input": ids, "mpp": mpp, "model": model}
        return self._submit_job(data)

    @catch_auth_exceptions
    def job_status(self, ticket_id: str, thumbnails: bool = False) -> Dict:
        """
        Gets the status of a job submitted to the Concentriq Embeddings endpoint.

        Parameters
        ----------
        ticket_id (str): The ticket ID of the submitted job.
        thumbnails (bool, optional): Whether the job is for thumbnails. Defaults to False.

        Returns
        -------
        Dict: The status of the job.
        """
        status_response = self.client.get_job_status(ticket_id, thumbnails=thumbnails)
        return status_response

    @catch_auth_exceptions
    def load_results(
        self,
        ticket_id: str,
        download_embeddings: bool = True,
        load_embeddings: bool = True,
        page_limit: int = 1000,
    ) -> Dict:
        """
        Fetches the results of a job from the cache or the Proscia endpoint.
        Optionally loads the embeddings.

        Parameters
        ----------
        ticket_id (str): The ticket ID of the submitted job.
        download_embeddings (bool, optional): Whether to download the embeddings. Defaults to True.
        load_embeddings (bool, optional): Whether to load the embeddings to device. Defaults to True.
        page_limit (int, optional): The number of results to fetch per page. Defaults to 1000.

        Returns
        -------
        dict: The results of the job.
        """
        cache_file = os.path.join(self.cache_dir, f"{ticket_id}.json")
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        offset = 0
        n_results = page_limit
        accumulated_embeddings = []
        while n_results == page_limit:
            if not os.path.exists(cache_file):
                from_cache = False
                embeddings = self.client.fetch_results(ticket_id, offset=offset, limit=page_limit).model_dump()
            else:
                from_cache = True
                with open(cache_file) as f:
                    embeddings = json.load(f)
            images = embeddings.get("images", [])
            n_results = len(images) if not from_cache else page_limit - 1
            for image in images:
                local_embedding_path = image.get("local_embedding_path", None)
                if download_embeddings:
                    local_embedding_path = self.download_embedding(image.get("embeddings_url", None))
                if load_embeddings and local_embedding_path is not None:
                    embedding = self.load_embedding(local_embedding_path)
                    image.update({"embedding": embedding})

                image.update({"local_embedding_path": local_embedding_path})
            accumulated_embeddings.extend(images)
            offset += page_limit
        embeddings["images"] = accumulated_embeddings
        with open(cache_file, "w") as f:
            # strip out the "thumbnail" key from the json
            cache_embeddings = {}
            cache_embeddings["images"] = [
                {k: v for k, v in t.items() if k != "embedding"} for t in embeddings["images"]
            ]
            json.dump(cache_embeddings, f)
        return embeddings

    def download_embedding(self, embeddings_url: str) -> str:
        """
        Downloads the embeddings to the cache from the Proscia endpoint.
        Only downloads if the embeddings are not already in the cache.

        Parameters
        ----------
        embeddings_url (str): The URL of the embeddings file.

        Returns
        -------
        dict: The embeddings.
        """
        if embeddings_url is None:
            return None
        fname = os.path.basename(embeddings_url).split("?")[0]
        path = os.path.join(self.cache_dir, fname)
        if not os.path.exists(path):
            res = requests.get(embeddings_url, timeout=600)
            with open(path, "wb") as f:
                f.write(res.content)

        return path

    def load_embedding(self, embeddings_path: str) -> dict:
        """
        Loads the embeddings from the cache.

        Parameters
        ----------
        embeddings_path (str): The URL of the embeddings file.

        Returns
        -------
        dict: The embeddings.
        """

        tensors = {}
        with safe_open(embeddings_path, framework="pt", device=self.device) as f:
            tensor_keys = f.keys()
            for k in tensor_keys:
                tensors[k] = f.get_tensor(k)

        return tensors

    def download_thumbnail(self, thumbnail_url: str) -> str:
        """
        Downloads the thumbnail image to local cache if it doesn't yet exist.

        Parameters
        ----------
        thumbnail_url (str): The URL of the thumbnail image.

        Returns
        -------
        str: The path to the thumbnail image.
        """
        if thumbnail_url is None:
            return None
        fname = os.path.basename(thumbnail_url).split("?")[0]
        local_path = os.path.join(self.cache_dir, fname)
        if not os.path.exists(local_path):
            res = requests.get(thumbnail_url, timeout=600)
            with open(local_path, "wb") as f:
                f.write(res.content)

        return local_path

    def load_thumbnail(self, local_path: str) -> np.ndarray:
        """
        Loads the thumbnail image from the cache.

        Parameters
        ----------
        local_path (str): path to the thumbnail image.

        Returns
        -------
        np.ndarray: The thumbnail image.
        """
        thumbnail = imageio.v2.imread(local_path)
        return thumbnail

    def get_embeddings(
        self,
        ticket_id: str,
        download_embeddings: bool = True,
        load_embeddings: bool = True,
        page_limit: int = 1000,
        polling_interval_seconds: int = 300,
    ) -> dict:
        """
        Gets the embeddings for a job.

        Parameters
        ----------
        ticket_id (str): The ticket ID of the submitted job.
        download_embeddings (bool, optional): Whether to download the embeddings. Defaults to True.
        load_embeddings (bool, optional): Whether to load the embeddings to device. Defaults to True.
        page_limit (int, optional): The number of results to fetch per page. Defaults to 1000.
        polling_interval_seconds (int, optional): The number of seconds to wait between polling for job status. Defaults to 300.

        Returns
        -------
        dict: The embeddings.
        """
        while True:
            status = None
            if not os.path.exists(os.path.join(self.cache_dir, f"{ticket_id}.json")):
                status = self.job_status(ticket_id)
            if status is None or status.status in ["completed", "completed with errors"]:
                embeddings = self.load_results(
                    ticket_id=ticket_id,
                    download_embeddings=download_embeddings,
                    load_embeddings=load_embeddings,
                    page_limit=page_limit,
                )
                return embeddings
            elif status.status == "failed":
                print(status)
                return None
            else:
                print(f"Waiting for job {ticket_id} to complete...")
                print(status)
                time.sleep(polling_interval_seconds)

    @catch_auth_exceptions
    def load_thumbnail_results(
        self,
        ticket_id: str,
        download_thumbnails: bool = True,
        load_thumbnails: bool = True,
        page_limit: int = 1000,
    ):
        """
        Fetches the results of a job from the cache or the Proscia endpoint.
        Optionally loads the thumbnails into memory.

        Parameters
        ----------
        ticket_id (str): The ticket ID of the submitted job.
        download_thumbnails (bool, optional): Whether to download the thumbnails. Defaults to True.
        load_thumbnails (bool, optional): Whether to load the thumbnails to device. Defaults to True.
        page_limit (int, optional): The number of results to fetch per page. Defaults to 1000.

        Returns
        -------
        dict: The results of the job.
        """
        cache_file = os.path.join(self.cache_dir, f"{ticket_id}.json")
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        offset = 0
        n_results = page_limit
        accumulated_thumbails = []
        while n_results == page_limit:
            if not os.path.exists(cache_file):
                from_cache = False
                thumbnails = self.client.fetch_results(
                    ticket_id, offset=offset, limit=page_limit, thumbnails=True
                ).model_dump()
            else:
                from_cache = True
                with open(cache_file) as f:
                    thumbnails = json.load(f)
            images = thumbnails.get("thumbnails", [])
            n_results = len(images) if not from_cache else page_limit - 1
            for image in images:
                local_thumbnail_path = image.get("local_thumbnail_path", None)
                if download_thumbnails:
                    local_thumbnail_path = self.download_thumbnail(image.get("thumb_url", None))
                if load_thumbnails and local_thumbnail_path is not None:
                    thumbnail = self.load_thumbnail(local_thumbnail_path)
                    image.update({"thumbnail": thumbnail})

                image.update({"local_thumbnail_path": local_thumbnail_path})
            accumulated_thumbails.extend(images)
            offset += page_limit
        thumbnails["thumbnails"] = accumulated_thumbails
        with open(cache_file, "w") as f:
            # strip out the "thumbnail" key from the json
            cache_thumbnails = {}
            cache_thumbnails["thumbnails"] = [
                {k: v for k, v in t.items() if k != "thumbnail"} for t in thumbnails["thumbnails"]
            ]
            json.dump(cache_thumbnails, f)
        return thumbnails

    def get_thumbnails(
        self,
        ticket_id: str,
        download_thumbnails: bool = True,
        load_thumbnails: bool = True,
        page_limit: int = 1000,
        polling_interval_seconds: int = 300,
    ):
        """
        Gets the thumbnails for a job.

        Parameters
        ----------
        ticket_id (str): The ticket ID of the submitted job.
        download_thumbnails (bool, optional): Whether to download the thumbnails. Defaults to True.
        load_thumbnails (bool, optional): Whether to load the thumbnails to device. Defaults to True.
        page_limit (int, optional): The number of results to fetch per page. Defaults to 1000.
        polling_interval_seconds (int, optional): The number of seconds to wait between polling for job status. Defaults to 300.

        Returns
        -------
        dict: The thumbnails.
        """
        while True:
            status = None
            if not os.path.exists(os.path.join(self.cache_dir, f"{ticket_id}.json")):
                status = self.job_status(ticket_id, thumbnails=True)
            if status is None or status.status in ["completed", "completed with errors"]:
                if status is not None:
                    print(status)
                thumbnails = self.load_thumbnail_results(
                    ticket_id=ticket_id,
                    download_thumbnails=download_thumbnails,
                    load_thumbnails=load_thumbnails,
                    page_limit=page_limit,
                )
                return thumbnails
            elif status.status == "failed":
                print(status)
                return None
            else:
                print(f"Waiting for job {ticket_id} to complete...")
                print(status)
                time.sleep(polling_interval_seconds)
