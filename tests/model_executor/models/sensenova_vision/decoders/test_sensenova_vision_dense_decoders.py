# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for SenseNova-Vision dense-image and structured-text decoders."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from vllm_omni.model_executor.models.sensenova_vision.decoders.camera_pose import parse_camera_pose
from vllm_omni.model_executor.models.sensenova_vision.decoders.dense_decoders import (
    decode_depth,
    decode_normal,
    decode_point_map,
    decode_segmentation,
)
from vllm_omni.model_executor.models.sensenova_vision.decoders.text_parsers import (
    parse_bbox,
    parse_keypoints,
    parse_points,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_decode_segmentation_binary():
    arr = ((np.arange(16 * 16).reshape(16, 16) * 16) % 256).astype(np.uint8)
    mask = decode_segmentation(arr)
    assert mask.shape == (16, 16)
    assert set(np.unique(mask)).issubset({0, 1})


def test_decode_segmentation_binary_threshold():
    arr = np.zeros((8, 8), dtype=np.uint8)
    arr[:, :4] = 200
    mask = decode_segmentation(arr, threshold=127)
    assert (mask[:, :4] == 1).all()
    assert (mask[:, 4:] == 0).all()


def test_decode_segmentation_rgb_classes():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[:, :4, 0] = 255  # red class 0
    rgb[:, 4:, 1] = 255  # green class 1
    class_define = [(255, 0, 0), (0, 255, 0)]
    mask = decode_segmentation(rgb, class_define=class_define)
    assert (mask[:, :4] == 0).all()
    assert (mask[:, 4:] == 1).all()


def test_decode_segmentation_rgb_black_background():
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[0, 0] = (255, 0, 0)
    class_define = [(255, 0, 0)]
    mask = decode_segmentation(rgb, class_define=class_define)
    assert mask[0, 0] == 0
    assert (mask[1:, :] == 1).all()
    assert (mask[0, 1:] == 1).all()


def test_decode_segmentation_accepts_pil(tmp_path):
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[2:, :] = (0, 255, 0)
    img = Image.fromarray(rgb)
    p = tmp_path / "mask.png"
    img.save(p)
    for source in (img, rgb, str(p)):
        mask = decode_segmentation(source, class_define=[(255, 0, 0), (0, 255, 0)])
        assert (mask[2:, :] == 1).all()


def test_decode_segmentation_invalid_palette():
    with pytest.raises(ValueError):
        decode_segmentation(np.zeros((4, 4, 3), dtype=np.uint8), class_define=[[1, 2]])


def test_decode_depth_grayscale():
    arr = np.zeros((8, 8, 3), dtype=np.uint8)
    arr[:, :, :] = 128
    depth = decode_depth(arr)
    assert depth.shape == (8, 8)
    assert np.allclose(depth, 128.0 / 255.0)


def test_decode_depth_2d_input():
    arr = np.full((6, 6), 64, dtype=np.uint8)
    depth = decode_depth(arr)
    assert depth.shape == (6, 6)
    assert np.allclose(depth, 64.0 / 255.0)


def test_decode_depth_resize():
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    arr[:] = 255
    depth = decode_depth(arr, size=(8, 8))
    assert depth.shape == (8, 8)
    assert np.allclose(depth, 1.0)

    img = Image.fromarray(arr)
    depth_pil = decode_depth(img, size=(8, 8))
    assert depth_pil.shape == (8, 8)
    assert np.allclose(depth_pil, 1.0)


def test_decode_normal_rgb():
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    arr[:, :, 2] = 255  # +z normal encoded as (0,0,255) in RGB
    normals = decode_normal(arr)
    assert normals.shape == (4, 4, 3)
    # +z: map (0,0,255)/255*2-1 = (-1,-1,1), then flip x -> (1,-1,1)
    assert np.allclose(normals[:, :, 0], 1.0)
    assert np.allclose(normals[:, :, 1], -1.0)
    assert np.allclose(normals[:, :, 2], 1.0)


def test_decode_normal_no_flip_x():
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    arr[:, :, 2] = 255
    normals = decode_normal(arr, flip_x=False)
    assert np.allclose(normals[:, :, 0], -1.0)


def test_decode_normal_resize():
    arr = np.full((4, 4, 3), 128, dtype=np.uint8)
    normals = decode_normal(arr, size=(2, 2))
    assert normals.shape == (2, 2, 3)

    img = Image.fromarray(arr)
    normals_pil = decode_normal(img, size=(2, 2))
    assert normals_pil.shape == (2, 2, 3)


def test_decode_point_map_float_array_passthrough():
    arr = np.random.default_rng(0).uniform(-1, 1, size=(4, 4, 3)).astype(np.float32)
    out = decode_point_map(arr)
    assert out.shape == (4, 4, 3)
    assert out.dtype == np.float32
    assert np.array_equal(out, arr)


def test_decode_point_map_uint8_scale():
    arr = np.full((4, 4, 3), 128, dtype=np.uint8)
    out = decode_point_map(arr)
    assert out.dtype == np.float32
    # (128/255*2-1) ~= 0.0039 (float32 tolerance)
    assert np.allclose(out, 128.0 / 255.0 * 2.0 - 1.0, atol=1e-6)


def test_decode_point_map_float_in_uint8_range_scaled():
    arr = np.full((4, 4, 3), 255.0, dtype=np.float32)
    out = decode_point_map(arr)
    assert np.allclose(out, 1.0)


def test_decode_point_map_invalid_shape():
    with pytest.raises(ValueError):
        decode_point_map(np.zeros((4, 4), dtype=np.float32))


def test_decode_point_map_pil_image():
    """A PIL point-map image is rescaled from [0, 255] to [-1, 1]."""
    arr = np.full((4, 4, 3), 128, dtype=np.uint8)
    out = decode_point_map(Image.fromarray(arr))
    assert out.shape == (4, 4, 3)
    assert out.dtype == np.float32
    assert np.allclose(out, 128.0 / 255.0 * 2.0 - 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Raw VAE-tensor (output_type="raw_tensor") input support, ~[-1, 1] space
# ---------------------------------------------------------------------------


def test_decode_depth_raw_float32_range_matches_uint8():
    """A raw VAE-space depth map decodes identically to its 8-bit equivalent.

    Raw float arrays in ``[-1, 1]`` (upstream ``output_raw_tensor=True``) are
    shifted back to byte scale first, so the Marigold-style ``mean/255`` math
    yields the same relative depth in ``[0, 1]``.
    """
    raw = np.zeros((8, 8, 3), dtype=np.float32)
    raw[:] = 128.0 / 255.0 * 2.0 - 1.0  # == byte 128
    depth = decode_depth(raw)
    assert depth.shape == (8, 8)
    assert np.allclose(depth, 128.0 / 255.0, atol=1e-6)


def test_decode_depth_raw_2d_input():
    """A grayscale raw depth map (HxW) is range-remapped before decoding."""
    raw = np.full((6, 6), 64.0 / 255.0 * 2.0 - 1.0, dtype=np.float32)
    depth = decode_depth(raw)
    assert depth.shape == (6, 6)
    assert np.allclose(depth, 64.0 / 255.0, atol=1e-6)


def test_decode_normal_raw_float32_matches_uint8():
    """A raw VAE-space normal map decodes to the same unit normals as 8-bit."""
    # +z encoded as byte (0,0,255); raw equivalent = (byte/255*2-1).
    # NB: raw -1.0 maps to byte 0, raw 1.0 to byte 255.
    raw = np.full((4, 4, 3), -1.0, dtype=np.float32)
    raw[:, :, 2] = 1.0  # -> byte (0, 0, 255)
    normals = decode_normal(raw)
    assert normals.shape == (4, 4, 3)
    # Same as the 8-bit path: (-1,-1,1) after remap, then flip x -> (1,-1,1).
    assert np.allclose(normals[:, :, 0], 1.0)
    assert np.allclose(normals[:, :, 1], -1.0)
    assert np.allclose(normals[:, :, 2], 1.0)


def test_decode_normal_raw_no_flip_x():
    raw = np.full((4, 4, 3), -1.0, dtype=np.float32)
    raw[:, :, 2] = 1.0
    normals = decode_normal(raw, flip_x=False)
    assert np.allclose(normals[:, :, 0], -1.0)


def test_decode_segmentation_raw_binary_threshold():
    """A raw binary mask (>0.5 in [-1,1] == >127 in [0,255]) threshold-decodes."""
    raw = np.full((8, 8), -1.0, dtype=np.float32)  # byte 0
    raw[:, :4] = 1.0  # byte 255
    mask = decode_segmentation(raw, threshold=127)
    assert (mask[:, :4] == 1).all()
    assert (mask[:, 4:] == 0).all()


def test_decode_segmentation_raw_rgb_classes():
    """Raw VAE-space RGB masks map to the same nearest-palette classes."""
    raw = np.zeros((8, 8, 3), dtype=np.float32)
    raw[:, :4, 0] = 1.0  # red class 0
    raw[:, 4:, 1] = 1.0  # green class 1
    class_define = [(255, 0, 0), (0, 255, 0)]
    mask = decode_segmentation(raw, class_define=class_define)
    assert (mask[:, :4] == 0).all()
    assert (mask[:, 4:] == 1).all()


def test_parse_bbox_single():
    text = "<p>dog</p><bbox>[0.1,0.2,0.8,0.9]</bbox>"
    parsed = parse_bbox(text)
    assert parsed == {"dog": [[0.1, 0.2, 0.8, 0.9]]}


def test_parse_bbox_multiple_and_clipping():
    text = "<p>cat</p><bbox>[-0.1,0.2,1.5,0.9]</bbox><bbox>[0.0,0.0,0.3,0.3]</bbox>"
    parsed = parse_bbox(text)
    assert parsed["cat"][0] == [0.0, 0.2, 0.999, 0.9]
    assert parsed["cat"][1] == [0.0, 0.0, 0.3, 0.3]


def test_parse_bbox_normalizes_category():
    text = "<p>person-1</p><bbox>[0.0,0.0,1.0,1.0]</bbox>"
    parsed = parse_bbox(text)
    assert "person 1" in parsed


def test_parse_bbox_empty():
    assert parse_bbox("") == {}
    assert parse_bbox("no tags here") == {}


def test_parse_points():
    text = "<p>traffic light</p><point>[0.4,0.6]</point>"
    parsed = parse_points(text)
    traffic_light = " ".join(parsed.keys()).lower()
    assert traffic_light == "traffic light"


def test_parse_points_multiple():
    text = "<p>pole</p><point>[0.1,0.2]</point><point>[0.3,0.4]</point>"
    parsed = parse_points(text)
    pole_key = " ".join(parsed.keys()).lower()
    assert parsed[pole_key] == [[0.1, 0.2], [0.3, 0.4]]


def test_parse_points_clipping():
    text = "<p>x</p><point>[1.2,-0.3]</point>"
    parsed = parse_points(text)
    key = " ".join(parsed.keys()).lower()
    assert parsed[key] == [[0.999, 0.0]]


def test_parse_keypoints_basic():
    text = (
        "<p>person</p><bbox>[0.0,0.0,0.5,0.5]</bbox>left shoulder<kpt>[0.1,0.2]</kpt>right shoulder<kpt>[0.3,0.4]</kpt>"
    )
    parsed = parse_keypoints(text)
    person_key = " ".join(parsed.keys()).lower()
    assert person_key == "person"
    instance = parsed[person_key][0]
    assert instance["bbox"] == [0.0, 0.0, 0.5, 0.5]
    assert instance["keypoints"]["left shoulder"] == [0.1, 0.2]
    assert instance["keypoints"]["right shoulder"] == [0.3, 0.4]


def test_parse_keypoints_invisible_and_ins():
    text = "<p>person</p><ins>1</ins>left eye<kpt>unvisible</kpt>right eye<kpt>[0.5,0.5]</kpt></ins>"
    parsed = parse_keypoints(text)
    person_key = " ".join(parsed.keys()).lower()
    kps = parsed[person_key][0]["keypoints"]
    # The official parser strips <ins> tags but keeps their text, so the
    # keypoint name includes the instance prefix "1".
    assert kps["1left eye"] == [-1.0, -1.0]
    assert kps["right eye"] == [0.5, 0.5]


def test_parse_camera_pose_frames():
    pose = (
        "<frame><quat>[0,0,0,1]</quat><offset>[100,0,0]</offset>"
        "<scale>200</scale></frame>"
        "<frame><quat>[1,0,0,0]</quat><offset>[0,-100,50]</offset>"
        "<scale>100</scale></frame>"
    )
    parsed = parse_camera_pose(pose)
    assert parsed is not None
    assert len(parsed["rotation"]) == 2
    assert len(parsed["translation"]) == 2
    # Values inside tags are milli-units: int/1000 for quat and offset, then
    # translation = offset * scale / 100.0 (official parser semantics).
    # offset [100, 0, 0] * scale 200 / 100.0 -> [0.2, 0.0, 0.0]
    assert parsed["translation"][0] == pytest.approx([0.2, 0.0, 0.0])
    # offset [0, -100, 50] * scale 100 / 100.0 -> [0.0, -0.1, 0.05]
    assert parsed["translation"][1] == pytest.approx([0.0, -0.1, 0.05])
    # quats are milli-units / 1000
    assert parsed["rotation"][1] == pytest.approx([0.001, 0.0, 0.0, 0.0])


def test_parse_camera_pose_no_frame_tags():
    pose = "<quat>[0,0,0,1]</quat><offset>[100,0,0]</offset><scale>100</scale>"
    parsed = parse_camera_pose(pose)
    assert parsed is not None
    # (100/1000) * 100 / 100.0 == 0.1
    assert parsed["translation"][0] == pytest.approx([0.1, 0.0, 0.0])


def test_parse_camera_pose_invalid():
    assert parse_camera_pose("no tags") is None
    assert parse_camera_pose("<offset>[0,0,0]</offset>") is None


def test_parse_camera_pose_non_string():
    with pytest.raises(TypeError):
        parse_camera_pose(123)
