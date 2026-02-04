from pydantic import BaseModel, Field


class EstimationResponse(BaseModel):
    job_cost: float = Field(..., title="Job Cost")
    credits_before_job: float = Field(..., title="Credits Before Job")
    credits_after_job: float = Field(..., title="Credits After Job")
    num_invalid_images: int | None = Field(None, title="Num Invalid Images")
    invalid_image_ids: list[int] | None = Field(None, title="Invalid Image Ids")


class SubmissionResponse(BaseModel):
    ticket_id: str = Field(..., title="Ticket Id")
    job_cost: float | None = Field(None, title="Job Cost")
    credits_before_job: float | None = Field(None, title="Credits Before Job")
    credits_after_job: float | None = Field(None, title="Credits After Job")


class ImageOutput(BaseModel):
    image_id: int = Field(..., title="Image Id")
    repository_id: int | None = Field(None, title="Repository Id")
    status: str = Field(..., title="Status")  # Assuming 'status' uses a fixed set of string values
    model: str | None = Field(None, title="Model")
    patch_size: int | None = Field(None, title="Patch Size")
    grid_rows: int | None = Field(None, title="Grid Rows")
    grid_cols: int | None = Field(None, title="Grid Cols")
    pad_height: int | None = Field(None, title="Pad Height")
    pad_width: int | None = Field(None, title="Pad Width")
    mpp: float | None = Field(None, title="Mpp")
    embeddings_url: str | None = Field(None, title="Embeddings Url")


class ThumbnailImageOutput(BaseModel):
    image_id: int = Field(..., title="Image Id")
    repository_id: int | None = Field(None, title="Repository Id")
    status: str = Field(..., title="Status")  # Assuming 'status' uses a fixed set of string values
    thumb_url: str | None = Field(None, title="Thumbnail Url")
    thumb_mpp: float | None = Field(None, title="Thumbnail Mpp")


class StatusResponse(BaseModel):
    status: str = Field(..., title="Status")
    progress: float | None = Field(None, title="Progress")
    finished: int | None = Field(None, title="Finished")
    failed: int | None = Field(None, title="Failed")
    queued: int | None = Field(None, title="Queued")
    processing: int | None = Field(None, title="Processing")


class JobOutput(BaseModel):
    images: list[ImageOutput] | None = Field(None, title="Images")


class ThumbnailsJobOutput(BaseModel):
    thumbnails: list[ThumbnailImageOutput] | None = Field(None, title="Thumbnail Images")
