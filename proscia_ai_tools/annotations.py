from collections.abc import Iterable

import cv2
import numpy as np
import pydantic
from pydantic import ConfigDict
from pydantic_extra_types.color import Color
from pydantic_xml import BaseXmlModel, attr, element


class AnnotationBounds(pydantic.BaseModel):
    """Object to represent the bounds of an annotation and provide serialization methods"""

    lower: list[float]
    size: list[float]

    def as_shapestring(self) -> str:
        return f"{self.lower[0]},{self.lower[1]} {self.size[0]},{self.size[1]}"

    model_config = ConfigDict(arbitrary_types_allowed=True)


class AnnotationShape(pydantic.BaseModel):
    """Object to hold a set of points used for representing annotations."""

    data: list[list[float]]
    """Array of x,y points"""

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> list[float]:
        return self.data[i]

    def as_shapestring(self) -> str:
        return " ".join([f"{p[0]},{p[1]}" for p in self.data])

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ConcentriqAnnotation(pydantic.BaseModel):
    """Data model for the Concentriq representation of an annotation."""

    imageId: int
    shape: str
    text: str
    bounds: AnnotationBounds
    points: AnnotationShape
    isNegative: bool
    color: Color
    isSegmenting: bool
    annotationClassId: int | None
    area: float
    """Area in micrometers. Defaults to zero if unspecified."""

    @pydantic.field_serializer("color")
    def serialize_color(self, color: Color) -> str:
        return color.as_hex()


def _hex_to_colorref(hex_color: str) -> int:
    hex_color = hex_color.strip("#")
    if len(hex_color) == 3:
        hex_color = "{}".format("".join(2 * c for c in hex_color))
    r = hex_color[0:2]
    g = hex_color[2:4]
    b = hex_color[4:6]
    # Alpha should be ignored for colorref.

    bgr = f"{b}{g}{r}"
    colorref = int(bgr, base=16)

    return colorref


class XMLVertex(BaseXmlModel):
    X: str = attr(name="X")
    Y: str = attr(name="Y")

    @pydantic.field_validator("*", mode="before")
    def is_str(cls, v):
        return str(v)


class XMLVertices(BaseXmlModel):
    Vertex: list[XMLVertex] = element(tag="Vertex")


class XMLRegion(BaseXmlModel):
    Text: str = attr(name="Text")
    NegativeROA: str = attr(name="NegativeROA")
    Type: str = attr(name="Type")
    Vertices: XMLVertices = element(tag="Vertices")
    Area: str = attr(name="Area")

    @pydantic.field_validator("NegativeROA", mode="before")
    def check_string_bools(cls, v):
        if isinstance(v, bool):
            v = str(int(v))
        if v not in ["0", "1"]:
            raise ValueError("NegativeROA must be a string representation of 0 or 1")  # noqa: TRY003
        return v

    @pydantic.field_validator("Type", mode="before")
    def type_to_aperio(cls, v):
        aperio_map = {
            "free": "0",
            "rect": "1",
            "ellipse": "2",
            "arrow": "3",
            "ruler": "4",
        }

        return aperio_map[v]


class XMLRegions(BaseXmlModel):
    Region: list[XMLRegion] = element(tag="Region")


class XMLAnnotation(BaseXmlModel):
    Name: str = attr(name="Name")
    ReadOnly: str = attr(name="ReadOnly")
    LineColor: str = attr(name="LineColor")
    Regions: XMLRegions = element(tag="Regions")

    @pydantic.field_validator("ReadOnly", mode="before")
    def check_string_bools(cls, v):
        if isinstance(v, bool):
            v = str(int(v))
        if v not in ["0", "1"]:
            raise ValueError("ReadOnly must be a string representation of 0 or 1")  # noqa: TRY003
        return v

    @pydantic.field_validator("LineColor", mode="before")
    def linecolor_to_colorref(cls, v):
        if isinstance(v, Color):
            v = v.as_hex()
        if v.startswith("#"):
            v = _hex_to_colorref(v)
        # value must be castable to an int.
        return str(int(v))


class Annotations(BaseXmlModel):
    MicronsPerPixel: str = attr(name="MicronsPerPixel")
    Annotation: list[XMLAnnotation] = element(tag="Annotation")


def pixel_to_viewport(point: tuple[float, float], img_height: int, img_width: int) -> tuple[float, float]:
    """Converts a pixel coordinate to a concentriq viewport.

    Parameters
    ----------
    point : Tuple[float, float]
        Point in pixel coordinates, (x,y).
    img_height : int
        Image height in pixels.
    img_width : int
        Image width in pixels.

    Returns
    -------
    Tuple[float, float]
        Point coordinates in Concentriq viewport space..
    """
    aspect_ratio = float(img_height) / float(img_width)
    width_conversion = img_width / 10000
    height_conversion = img_height / (10000 * aspect_ratio)
    x = point[0] / width_conversion
    y = point[1] / height_conversion
    return (x, y)


def viewport_to_pixel(point: tuple[float, float], img_height: int, img_width: int) -> tuple[float, float]:
    """Converts a concentriq viewport coordinate to a pixel coordinate.

    Parameters
    ----------
    point : Tuple[float, float]
        Point in Concentriq viewport coordinates, (x,y).
    img_height : int
        Image height in pixels.
    img_width : int
        Image width in pixels.

    Returns
    -------
    Tuple[float, float]
        Point coordinates in pixel space.
    """
    aspect_ratio = float(img_height) / float(img_width)
    width_conversion = img_width / 10000
    height_conversion = img_height / (10000 * aspect_ratio)
    x = point[0] * width_conversion
    y = point[1] * height_conversion
    return (x, y)


def _contour_to_annotation_points(
    contour: Iterable[Iterable[int]], img_width: int, img_height: int
) -> tuple[AnnotationShape, AnnotationBounds, float]:
    """Returns pixel information for a free annotation created from a contour.

    .. note::
        If set to img_width=10000 and img_height=10000,
        pixel-to-viewport mapping is effectively disabled.
        This is useful if downstream applications need to stay in pixel space.

    This function returns partial information needed to create a FreeAnnotation.
    """
    viewport_points = [pixel_to_viewport(p, img_height, img_width) for p in contour]
    viewport_array = np.array(viewport_points)

    viewport_min = np.min(viewport_array, axis=0)
    viewport_size = np.max(viewport_array, axis=0) - viewport_min
    viewport_shape = AnnotationShape(data=viewport_array.tolist())
    viewport_bounds = AnnotationBounds(lower=viewport_min.tolist(), size=viewport_size.tolist())

    # Scaling to mpp in the platform
    mpp_scale = (img_width / 10000) ** 2
    area = cv2.contourArea(viewport_array.astype(np.int32)) * mpp_scale
    return viewport_shape, viewport_bounds, area


def concentriq_annotation_to_xml(
    annotations: list[ConcentriqAnnotation],
) -> Annotations:
    xml_annotations = []
    for annot in annotations:
        vertices = XMLVertices(Vertex=[XMLVertex(X=p[0], Y=p[1]) for p in annot.points.data])
        region = XMLRegion(
            Text=annot.text,
            NegativeROA=annot.isNegative,
            Type=annot.shape,
            Vertices=vertices,
            Area=str(annot.area),
        )
        xml_annot = XMLAnnotation(
            Name=annot.text,
            ReadOnly="0",
            LineColor=annot.color,
            Regions=XMLRegions(Region=[region]),
        )
        xml_annotations.append(xml_annot)

    annotation_doc = Annotations(MicronsPerPixel="", Annotation=xml_annotations)

    return annotation_doc


class UnprocessableContour(Exception): ...


def unique_unsorted(array: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Performs a unique on an array without changing the order - keeps first detection.

    Parameters
    ----------
    array : np.ndarray
        Array to perform unique.
    axis : int, optional
        Axis along which to perform unique, by default None.

    Returns
    -------
    np.ndarray
        Uniqued array, unsorted.
    """
    uniq, index = np.unique(array, axis=axis, return_index=True)
    return uniq[index.argsort()]


def resize_contour(contour: np.ndarray, mpp_resize_ratio: float) -> np.ndarray:
    """Resizes contour using mpp_resize_ratio

    Parameters
    ----------
    contour : np.ndarray
        Array of points in contour
    mpp_resize_ratio : float
        Resize ratio calculated by mpp, formula is `original_mpp / desired_mpp`

    Returns
    -------
    np.ndarray
        Resized contour
    """
    resized_contour = unique_unsorted(np.round(contour * mpp_resize_ratio), axis=0)
    return resized_contour.astype(contour.dtype)


def mask2contours(mask: np.ndarray) -> np.ndarray:
    """Converts a mask to a contour.

    Parameters
    ----------
    mask : np.ndarray
        Mask to convert to contours.

    Returns
    -------
    np.ndarray
        Contours of mask.
    """
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def create_contour_annotation(
    contour: Iterable[Iterable[float]],
    image_id: int,
    text: str,
    color: Color,
    is_negative=True,
    img_width: int | None = None,
    img_height: int | None = None,
    resize_ratio: float | None = None,
    annotation_class_id: int | None = None,
) -> ConcentriqAnnotation:
    """Create a free annotation from a contour.

    Parameters
    ----------
    contour : Iterable[Iterable[float]]
        An iterable containing contours of X,Y pixel pairs.
        This may be a numpy.ndarray, but in all cases, contours are converted
        to an ndarray and extra axes are squeezed out.
    image_id : int
        Image ID to which the contours belong.
    text : str
        Text name of the annotation.
    img_width : int
        Width of the WSI object on Concentriq.
        **This is set to 10,000 if not specified, which effectively disables
        pixel-to-Concentriq viewport mapping**.
    img_height : int
        Height of the WSI object on Concentriq.
        **This is set to 10,000 if not specified, which effectively disables
        pixel-to-Concentriq viewport mapping**.
    color : Color
        General color representation that will be converted to Concentriq-compatible hex.
        This can be either a string name or a raw hex code.
        See `pydantic.color` docs for details.
    resize_ratio : Optional[float]
        Ratio at which to resize the contour pixels in order to make them match
        the destination image on Concentriq.
        This is helpful if the contours are derived from a thumbnail or smaller-resolution image,
        as these contour pixels need to be resized back to the native resolution of the WSI.
        This value, if used, should be set to:
            resize_ratio = contour_mpp / native_mpp
        where the contour MPP is the microns-per-pixel at which the contour was generated (IE the source)
        and the native_mpp is the microns-per-pixel of the image on Concentriq (IE the destination)
    annotation_class_id : Optional[int]

    Returns
    -------
    ConcentriqAnnotation :
        A payload which can be POSTed to the Concentriq API to create an annotation
        from the provided contour
    """
    if (img_width is None) ^ (img_height is None):
        raise AttributeError("If specified, img_width and img_height must both be supplied.")  # noqa: TRY003
    # If not specified, set img_width and height to 10,000, which disables contour pixels-to-viewport mapping.
    if img_width is None and img_height is None:
        img_width = 10000
        img_height = 10000

    # Check contour dimensionality.
    resized_contour = np.array(contour).squeeze()
    resized_contour = np.atleast_2d(resized_contour)  # Handle 1-pt contours.

    # resized_contour = _decimate_contour(resized_contour)

    if resize_ratio is not None:
        resized_contour = resize_contour(resized_contour, resize_ratio)
    if len(resized_contour) < 4:
        raise UnprocessableContour("Contour had too few points, possibly due to non-unique points being dropped.")  # noqa: TRY003
    shape, bounds, area = _contour_to_annotation_points(resized_contour, img_height=img_height, img_width=img_width)
    cx_annotation = ConcentriqAnnotation(
        imageId=image_id,
        shape="free",
        text=text,
        bounds=bounds,
        points=shape,
        isNegative=is_negative,
        color=color,
        isSegmenting=False,
        area=area,
        annotationClassId=annotation_class_id,
    )
    return cx_annotation
