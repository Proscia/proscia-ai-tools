import io
import json
import logging
import time
from functools import wraps
from typing import Dict, List, Optional

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from requests.auth import HTTPBasicAuth
from requests.models import HTTPError
from urllib3.util.retry import Retry

from proscia_ai_tools import utils
from proscia_ai_tools.annotations import ConcentriqAnnotation, create_contour_annotation, mask2contours


class GetException(Exception):
    """Exception raised when get request fails."""

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


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

        retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
        self.session = requests.Session()
        self.session.mount("http://", HTTPAdapter(max_retries=retries))
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        self.refresh_client_token()

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
                self.refresh_client_token()
                return func(self, *args, **kwargs)

        return wrapper

    def _get_token(self) -> None:
        url = f"{self.endpoint}/v3/auth/token"
        payload = {}
        response = requests.request(
            "POST", url, auth=HTTPBasicAuth(username=self.username, password=self.password), data=payload
        )
        self.token = response.json().get("token", None)

    def refresh_client_token(self):
        self._get_token()
        self.session.headers.update({"Authorization": f"Bearer {self.token}"})

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

    def get_annotations(
        self, image_ids: List[int], annotationId: Optional[List[int]] = None, text: Optional[List[str]] = None
    ):
        """Get annotations from Concentriq

        Parameters:
        ----------
        image_ids: "list" of image ids in Concentriq
        annotationId: "list" of annotation ids in Concentriq. This is an optional filter.
        text: "list" of annotation texts in Concentriq. This is an optional filter.

        Returns:
        -------
        'dict' with annotations data.
        """
        filters = {"imageId": image_ids}
        if annotationId is not None:
            filters["annotationId"] = annotationId
        if text is not None:
            filters["text"] = text

        annotations_request_filters = {"filters": json.dumps(filters)}
        annotations_response_dict = self.paginated_get_query(
            f"{self.endpoint}/annotations", params=annotations_request_filters
        )

        return annotations_response_dict["data"]

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
    def create_annotation(self, annotation: ConcentriqAnnotation) -> Dict:
        """Creates an annotation.

        Parameters
        ----------
        text (str): The name of the annotation.
        color (str): The color of the annotation class in hex.

        Returns
        -------
        dict: The created annotation
        """
        resp = None
        url = f"{self.endpoint}/annotations"
        data = {
            "imageId": annotation.imageId,
            "annotationClassId": annotation.annotationClassId,
            "shape": "free",
            "text": annotation.text,
            "shapeString": annotation.points.as_shapestring(),
            "captureBounds": annotation.bounds.as_shapestring(),
            "color": annotation.color.as_hex(),
            "isNegative": annotation.isNegative,
        }
        try:
            resp = self.session.post(url, json=data)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as error:
            self.log_http_error(
                error,
                req={
                    "path": url,
                    "data": data,
                },
                resp=resp.text if resp is not None else None,
            )

    @catch_auth_exceptions
    def create_annotation_class(
        self, annotation_class_name: str, description: str = "", color: str = "#000000"
    ) -> Dict:
        """Creates an annotation class.

        Parameters
        ----------
        annotation_class_name (str): The name of the annotation class.
        description (str): The description of the annotation class.
        color (str): The color of the annotation class in hex.

        Returns
        -------
        dict: The created annotation class.
            example: {"name":"zzz","description":"this description","color":"#011FFF",
            "createdBy":46,"lastUpdatedBy":46,"id":89,
            "createdAt":"2025-01-06T00:48:15.555Z",
            "lastUpdatedAt":"2025-01-06T00:48:15.555Z",
            "sysPeriod":"[\"2025-01-06 00:48:15.555399+00\",)"}
        """
        resp = None
        url = f"{self.endpoint}/v3/annotationClasses"
        data = {"name": annotation_class_name, "description": description, "color": color}
        try:
            resp = self.session.post(url, json=data)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as error:
            self.log_http_error(
                error,
                req={
                    "path": url,
                    "data": data,
                },
                resp=resp.text if resp is not None else None,
            )

    @catch_auth_exceptions
    def add_annotation_class_to_repo(self, annotation_class_id: int, repo_id: int):
        """Adds an annotation class to a repository.

        Parameters
        ----------
        annotation_class_id (int): The ID of the annotation class.
        repo_id (int): The ID of the repository.
        """
        resp = None
        url = f"{self.endpoint}/v3/annotationClasses/{annotation_class_id}/imageSets/{repo_id}"
        try:
            resp = self.session.post(url)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as error:
            self.log_http_error(
                error,
                req={"path": url},
                resp=resp.text if resp is not None else None,
            )

    @catch_auth_exceptions
    def assign_annotation_to_class(self, annotation_id: int, annotation_class_id: int):
        """Assign an annotation to an annotation class.

        Parameters
        ----------
        annotation_id (int): The ID of the annotation.
        annotation_class_id (int): The ID of the annotation class.
        """
        resp = None
        url = f"{self.endpoint}/annotations"
        data = {"annotations": [{"annotationId": annotation_id, "annotationClassId": annotation_class_id}]}
        try:
            resp = self.session.patch(url, json=data)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as error:
            self.log_http_error(
                error,
                req={"path": url},
                resp=resp.text if resp is not None else None,
            )

    def insert_annotations_from_mask(
        self,
        image_id: int,
        mask: np.ndarray,
        mask_mpp: float,
        annotation_name: str = "",
        color: str = "#0000FF",
        is_negative: bool = False,
        annotation_class_id: Optional[int] = None,
    ) -> Dict:
        """Converts a mask image into annotations and inserts them into the Concentriq LS platform.

        Parameters
        ----------
        image_id (int): The ID of the image.
        mask (np.ndarray): The mask to overlay.
        mask_mpp (float): The microns per pixel of the mask.
        annotation_name (str): The name of the annotation to be displayed in the platform.
        color (str): The color of the annotation.
        is_negative (bool): Whether the annotation is negative.
        annotation_class_id (Optional[int]): The ID of the annotation class.

        Returns
        -------
        dict: The created annotations.
        """
        image_data = self.get_image_data(image_id)
        image_mpp = image_data["mppx"]
        w, h = image_data.get("imgWidth"), image_data.get("imgHeight")

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
                img_height=h,
                img_width=w,
                annotation_class_id=annotation_class_id,
            )
            annotations.append(self.create_annotation(annotation))

        return annotations
