from typing import List, Optional

from pydantic import BaseModel, Field


class EstimationResponse(BaseModel):
    job_cost: float = Field(..., title="Job Cost")
    credits_before_job: float = Field(..., title="Credits Before Job")
    credits_after_job: float = Field(..., title="Credits After Job")
    num_invalid_images: Optional[int] = Field(None, title="Num Invalid Images")
    invalid_image_ids: Optional[List[int]] = Field(None, title="Invalid Image Ids")


class SubmissionResponse(BaseModel):
    ticket_id: str = Field(..., title="Ticket Id")
    job_cost: Optional[float] = Field(None, title="Job Cost")
    credits_before_job: Optional[float] = Field(None, title="Credits Before Job")
    credits_after_job: Optional[float] = Field(None, title="Credits After Job")


class ImageOutput(BaseModel):
    image_id: int = Field(..., title="Image Id")
    repository_id: Optional[int] = Field(None, title="Repository Id")
    status: str = Field(..., title="Status")  # Assuming 'status' uses a fixed set of string values
    model: Optional[str] = Field(None, title="Model")
    patch_size: Optional[int] = Field(None, title="Patch Size")
    grid_rows: Optional[int] = Field(None, title="Grid Rows")
    grid_cols: Optional[int] = Field(None, title="Grid Cols")
    pad_height: Optional[int] = Field(None, title="Pad Height")
    pad_width: Optional[int] = Field(None, title="Pad Width")
    mpp: Optional[float] = Field(None, title="Mpp")
    embeddings_url: Optional[str] = Field(None, title="Embeddings Url")


class ThumbnailImageOutput(BaseModel):
    image_id: int = Field(..., title="Image Id")
    repository_id: Optional[int] = Field(None, title="Repository Id")
    status: str = Field(..., title="Status")  # Assuming 'status' uses a fixed set of string values
    thumb_url: Optional[str] = Field(None, title="Thumbnail Url")
    thumb_mpp: Optional[float] = Field(None, title="Thumbnail Mpp")


class StatusResponse(BaseModel):
    status: str = Field(..., title="Status")
    progress: Optional[float] = Field(None, title="Progress")
    finished: Optional[int] = Field(None, title="Finished")
    failed: Optional[int] = Field(None, title="Failed")
    queued: Optional[int] = Field(None, title="Queued")
    processing: Optional[int] = Field(None, title="Processing")


class JobOutput(BaseModel):
    images: Optional[List[ImageOutput]] = Field(None, title="Images")


class ThumbnailsJobOutput(BaseModel):
    thumbnails: Optional[List[ThumbnailImageOutput]] = Field(None, title="Thumbnail Images")
