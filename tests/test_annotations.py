import pytest
from pydantic_extra_types.color import Color

from utils import annotations


def test_valid_short_hex_color():
    assert annotations._hex_to_colorref("#123") == 3351057


def test_valid_long_hex_color():
    assert annotations._hex_to_colorref("#AABBCC") == 13417386


def test_invalid_hex_color():
    with pytest.raises(ValueError):
        annotations._hex_to_colorref("invalid_hex")


def test_non_hex_characters():
    with pytest.raises(ValueError):
        annotations._hex_to_colorref("#XYZ")


def test_XMLVertex_validator():
    vertex = annotations.XMLVertex(X=1, Y=2)
    assert vertex.X == "1"
    assert vertex.Y == "2"


def test_XMLRegion_validator():
    vertices = annotations.XMLVertices(Vertex=[annotations.XMLVertex(X="1", Y="2")])
    region = annotations.XMLRegion(Text="Text", NegativeROA=True, Type="free", Vertices=vertices, Area="0")
    assert region.NegativeROA == "1"
    assert region.Type == "0"
    region = annotations.XMLRegion(Text="Text", NegativeROA=True, Type="free", Vertices=vertices, Area="0")
    with pytest.raises(ValueError):
        region = annotations.XMLRegion(
            Text="Text", NegativeROA="random_input", Type="free", Vertices=vertices, Area="0"
        )


def test_XMLAnnotation_validator():
    vertices = annotations.XMLVertices(Vertex=[annotations.XMLVertex(X="1", Y="2")])
    region = annotations.XMLRegion(Text="Text", NegativeROA=True, Type="free", Vertices=vertices, Area="0")
    regions = annotations.XMLRegions(Region=[region])
    annotation = annotations.XMLAnnotation(Name="name", ReadOnly=True, LineColor="#123", Regions=regions)
    assert annotation.ReadOnly == "1"
    assert annotation.LineColor == "3351057"
    c = Color("#123")
    annotation = annotations.XMLAnnotation(Name="name", ReadOnly=True, LineColor=c, Regions=regions)
    assert annotation.LineColor == "3351057"
