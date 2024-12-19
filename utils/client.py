import io
import json
import logging
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
from utils import utils
from utils.annotations import Annotations, concentriq_annotation_to_xml, create_contour_annotation, mask2contours


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
                    local_embedding_path = self.download_embedding(image["embeddings_url"])
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
                    local_thumbnail_path = self.download_thumbnail(image["thumb_url"])
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


class ConcentriqLSClient:
    DEFAULT_PAGINATION_ITEM_COUNT = 500

    def __init__(self, url: str, email: str, password: str, pagination_info=None):
        """Client for the Concentriq LS api.

        Parameters:
        -----------
        url: "str" of the Concentriq LS endpoint
        email: "str"
        password: "str"  #  pragma: allowlist secret
        pagination_info: Optional[dict]
            Key: "itemsPerPage" Val: 'int rows per page
                Default: 500

            Key: "sortBy" Val: "str|List[str]" possible sorts "created","lastModified","name","size"
                Default: "created"

            Key: "descending" Val: "bool" sort order
                    Default: "False"

                Key: "query_sleep" Val: "float" sleep time between iterative queries in seconds
                    Default: 0.2

                Note: When paginated queries are involved user has to be aware of
                affect of sorting configuration default is robust to an extent to
                prevent any duplicates
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        if pagination_info is None:
            pagination_info = {}
        self.endpoint = f"{url}/api"
        self.username = email
        self.password = password
        self.refresh_session()

        pagination_info["page"] = 1
        pagination_info["descending"] = pagination_info.get("descending", False)
        self._items_per_page = pagination_info.get("itemsPerPage", self.DEFAULT_PAGINATION_ITEM_COUNT)
        pagination_info["itemsPerPage"] = self._items_per_page
        pagination_info["sortBy"] = pagination_info.get("sortBy", ["created"])
        self.pagination_info = pagination_info
        self.query_sleep = pagination_info.get("query_sleep", 0.2)

    def catch_auth_exceptions(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except HTTPError:
                self.refresh_session()
                return func(self, *args, **kwargs)

        return wrapper

    def _get_token(self) -> None:
        url = f"{self.endpoint}/v3/auth/token"
        payload = {}
        response = requests.request(
            "POST", url, auth=HTTPBasicAuth(username=self.username, password=self.password), data=payload
        )
        self.token = response.json().get("token", None)

    def refresh_session(self):
        session = requests.Session()
        self._get_token()
        self.session = session
        self.session.headers.update({"token": self.token})

    @catch_auth_exceptions
    def create_overlay(self, image_id: int, overlay_name: str, module_id=1) -> Dict:
        """Issue a HTTP POST request to create an overlay object for an image resource
        Parameters:
        -----------
        image_id: "int" of the Concentriq image id for which the overlay is
            being uploaded.
        overlay_name: "str" of overlay name
        module_id: "int" of the module id for the overlay

        Returns:
        --------
        'Dict' with created overlay object

        Raises:
        -------
        Exception in case of HTTP POST request failure.
        """
        overlay_post_request_dict = {"name": overlay_name, "moduleId": module_id}
        url = f"{self.endpoint}/images/{int(image_id)}/overlays"
        overlay_post_response = self.session.post(url, json=overlay_post_request_dict)
        overlay_post_response.raise_for_status()
        overlay_post_response = overlay_post_response.json()
        return overlay_post_response["data"]

    @catch_auth_exceptions
    def sign_overlay_s3_url(self, overlay_id: int) -> str:
        """Signs an S3 URL for uploading an overlay.

        Parameters
        ----------
        overlay_id (int): The ID of the overlay.

        Returns
        -------
        str: The signed URL.
        """
        response = self.session.get(f"{self.endpoint}/sign_s3_overlay/{overlay_id}")
        response.raise_for_status()
        signed_url = response.json()["signed_request"]
        return signed_url

    @catch_auth_exceptions
    def upload_overlay(self, signed_url: str, overlay_data: bytes) -> None:
        """Uploads an overlay to S3.

        Parameters
        ----------
        signed_url (str): The signed URL.
        overlay_data (bytes): The overlay data.
        """
        overlay_data.seek(0)
        response = self.session.put(signed_url, data=overlay_data, headers={"Authorization": None})
        response.raise_for_status()

    def insert_heatmap_overlay(self, image_id: int, heatmap: np.ndarray, result_name: str) -> None:
        """Inserts a heatmap overlay into the Concentriq LS platform.

        Parameters
        ----------
        image_id (int): The ID of the image.
        heatmap (np.ndarray): The heatmap to overlay.
        result_name (str): The name of the overlay to be displayed in the platform.
        """
        img = utils.create_overlay(heatmap)
        image_data = utils.image_as_bytes(img)
        overlay = self.create_overlay(image_id, result_name)
        signed_url = self.sign_overlay_s3_url(overlay["id"])
        self.upload_overlay(signed_url, io.BytesIO(image_data))

    @catch_auth_exceptions
    def upload_xml_annotations(self, image_id: int, annotations: Annotations) -> None:
        """Creates annotations using the xml annotation import endpoint.

        Parameters
        ----------
        image_id (int): The ID of the image.
        annotations (Annotations): The annotations to upload.
        """
        resp = None
        url = f"{self.endpoint}/images/{image_id}/annotations/import"
        try:
            files = {
                "file": annotations.to_xml(encoding="UTF-8"),
            }
            resp = self.session.post(url, files=files)
            resp.raise_for_status()
        except requests.exceptions.HTTPError as error:
            self.log_http_error(
                error,
                req={
                    "path": url,
                    "data": annotations.to_xml(pretty_print=True, encoding="UTF-8").decode(),
                },
                resp=resp.text if resp is not None else None,
            )

    @catch_auth_exceptions
    def get_image_data(self, image_id: int):
        """Get image data for an image resource in Concentriq

        Parameters:
        ----------
        image_id: "int" image id in Concentriq

        Returns:
        --------
        'dict' with image data.

        Raises:
        -------
        Exception in case of HTTP GET request failure.
        """
        try:
            url = f"{self.endpoint}/images/{image_id}"
            image_data_get_response = self.session.get(url)
            image_data_get_response.raise_for_status()
        except requests.exceptions.HTTPError as error:
            self.log_http_error(
                error,
                req={
                    "path": url,
                    "params": None,
                },
                resp=image_data_get_response.text,
            )
            raise

        image_data_get_response_dict = image_data_get_response.json()
        if "error" in image_data_get_response_dict:
            raise GetException(image_data_get_response_dict["error"])
        else:
            return image_data_get_response_dict["data"]

    @catch_auth_exceptions
    def get_repo_data(self, target_repo_id: str) -> Dict:
        """Get metadata for requested repo in Concentriq
        Parameters:
        ----------
        target_repo_id: "str" repo id in Concentriq

        Returns:
        --------
        'dict' with repo data. Refer to Concentriq for further documentation

        Raises:
        -------
        Exception in case of HTTP GET request failure.
        """
        try:
            url = f"{self.endpoint}/imageSets/{int(target_repo_id)}"
            repo_data_get_response = self.session.get(url)
            repo_data_get_response.raise_for_status()
        except requests.exceptions.HTTPError as error:
            self.log_http_error(
                error,
                req={
                    "path": url,
                    "params": None,
                },
                resp=repo_data_get_response.text,
            )
            raise

        repo_data_get_response_dict = repo_data_get_response.json()
        if "error" in repo_data_get_response_dict:
            raise GetException(repo_data_get_response_dict["error"])
        else:
            return repo_data_get_response_dict["data"]

    @catch_auth_exceptions
    def paginated_get_query(self, url: str, params: dict) -> Dict:
        """Method to perform series of queries with incremental page numbers to
        retreive all entities with default GET retrivel limited with pagination

        NOTE: if params contain a pagination defined call will be made only using
        set pagination.

        Parameters:
        -----------
        url: "str" of the endpoint to be queried
        params: 'dict' of query parameters to be used in the query

        Returns:
        --------
        'dict' of the response from the query
        """
        if "pagination" not in params:
            result = {}
            i = 0
            while True:
                i = i + 1
                self.pagination_info["page"] = i
                params["pagination"] = json.dumps(self.pagination_info)
                try:
                    response = self.session.get(url, params=params)
                    print(response.json())
                    response.raise_for_status()
                except requests.exceptions.HTTPError as error:
                    self.log_http_error(
                        error,
                        req={
                            "path": url,
                            "params": params,
                        },
                        resp=response.text,
                    )
                    raise
                response_json = response.json()
                if i == 1:
                    entity = next(iter(response_json["data"].keys()))
                for each_entity in response_json["data"][entity]:
                    result[each_entity["id"]] = each_entity
                # Break out if page is last by comparing out image count with row images.
                if len(response_json["data"][entity]) < self._items_per_page:
                    break
                time.sleep(self.query_sleep)
            return_json = {"data": {entity: list(result.values())}}
            return return_json
        else:
            self.logger.info("Preset pagination detected using it to perform a single query!")
            try:
                response = self.session.get(url, params=params)
                response.raise_for_status()
            except requests.exceptions.HTTPError as error:
                self.log_http_error(
                    error,
                    req={
                        "path": url,
                        "params": params,
                    },
                    resp=response.text,
                )
                raise
            response_json = response.json()
            return response_json

    def get_images_in_a_repo(self, target_repo_id=None, custom_filter=None):
        """Issue a HTTP GET request to fetch data in a target repo
        Parameters:
        -----------
        target_repo_id: "str" of target repo id.
        custom_filter: 'dict' dict defining filters to be applied to retreive images
                        custom filters can be used to define the filtering mechanism involving any entity to filter the
                        contents of repository to be retrieved.

                        In the dict key refers to entity to be used for filter and value would refer to value to be applied
                        for filtering.

                        Possible/tested filters:
                        -----------------
                            Key: imageSetId
                            Val: 'list(repo_ids)' # to retreive images from endpoint from given repo_id's

                            Key: fields
                            Val: 'dict'
                                Key: field_id (int) field_id to be used to filter
                                Val: 'dict'
                                    Key: "ContentType"
                                    Val: 'str' (Type of data whether it is string or float or int)

                                    Key: "values"
                                    Val: 'list' of list of values of above said ContentType to be used
                                            to apply filter

                        For detailed list of possible filters and definitions:

                                https://<concentriq-ls-url>/api-documentation/#Image-GetImages

                        Example:
                        -------
                        custom_filter to be used to get all the images in a repo with repo_id as '118'. this is what is created when
                        target_repo_id is provided. same can be acheived with following custom_filter

                            custom_filter = {'imageSetId':[118]}

                        custom_filter to be used to get images in repo with repo_id as '118' with metadata field 'AnalysisStatus' with
                        field-id as 'field_id' with metadata value as 'Analyzing' for  field by using following custom filter

                            custom_filter = {'imageSetId':[118],
                                            'fields':{
                                            field_id:{
                                            "contentType": "String",
                                            "values": ['Analyzing']
                                            }
                                            }}

        Returns:
        --------
        'dict' with key "images"
                    value "list" of images in the target repo

        Raises:
        -------
        Exception in case of HTTP GET request failure.
        """

        if custom_filter is None:
            images_request_filters = {"filters": json.dumps({"imageSetId": [target_repo_id]})}
        else:
            images_request_filters = {"filters": json.dumps(custom_filter)}

        images_response_dict = self.paginated_get_query(f"{self.endpoint}/images", images_request_filters)
        return images_response_dict["data"]

    def log_http_error(self, error: requests.exceptions.HTTPError, req=None, resp=None):
        """
        Method to log unsuccessfull http requests.
        Logs include request url, request headers, request body, response headers, response body
        Parameters:
        ----------
        error: 'requests.exceptions.HTTPError' object
        req: 'dict' with request details
        resp: 'str' response text
        """
        # Sanitize request data
        sanitized_request = {k: (v if k != "Authorization" else "[REDACTED]") for k, v in req.items()} if req else None

        error_message = {
            "status_code": error.response.status_code,
            "request": sanitized_request,
            "response": resp,
        }
        clean_error_message = {k: v for k, v in error_message.items() if v is not None}
        self.logger.error("%s", clean_error_message)

    @catch_auth_exceptions
    def create_xml_annotations(self, image_id: int, annotations: Annotations):
        """Creates annotations using the xml annotation import endpoint."""
        resp = None
        url = f"{self.endpoint}/images/{image_id}/annotations/import"
        try:
            files = {
                "file": annotations.to_xml(encoding="UTF-8"),
            }
            resp = self.session.post(url, files=files)
            resp.raise_for_status()
        except requests.exceptions.HTTPError as error:
            self.log_http_error(
                error,
                req={
                    "path": url,
                    "data": annotations.to_xml(pretty_print=True, encoding="UTF-8").decode(),
                },
                resp=resp.text if resp is not None else None,
            )

    def insert_annotaions_from_mask(
        self,
        image_id: int,
        mask: np.ndarray,
        mask_mpp: float,
        annotation_name: str = "",
        color: str = "#0000FF",
        is_negative: bool = False,
    ):
        """Converts a mask image into annotations and inserts them into the Concentriq LS platform.

        Parameters
        ----------
        image_id (int): The ID of the image.
        mask (np.ndarray): The mask to overlay.
        mask_mpp (float): The microns per pixel of the mask.
        annotation_name (str): The name of the annotation to be displayed in the platform.
        color (str): The color of the annotation.
        is_negative (bool): Whether the annotation is negative.
        """
        image_data = self.get_image_data(image_id)
        image_mpp = image_data["mppx"]

        contours = mask2contours(mask)
        annotations = []
        for contour in contours:
            annotation = create_contour_annotation(
                contour=contour,
                image_id=image_id,
                text=annotation_name,
                color=color,
                is_negative=is_negative,
                resize_ratio=mask_mpp / image_mpp,
            )

            annotations.append(annotation)
        annotation_payload = concentriq_annotation_to_xml(annotations)
        self.create_xml_annotations(image_id, annotation_payload)
