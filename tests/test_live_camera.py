import numpy as np
import torch
import av
import threading

from live_camera import LiveCameraStats, build_rtc_configuration, camera_media_stream_constraints, make_video_frame_callback, process_camera_frame, resize_camera_frame


class FakeModel:
    def __init__(self):
        self.cls_thr = 0.1
        self.nms_thr = 0.5
        self.residual_cnn = FakeResidualCNN()
        self.calls = 0

    def eval(self):
        return self

    def __call__(self, tensor):
        self.calls += 1
        return [{
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float32),
            "scores": torch.tensor([0.9], dtype=torch.float32),
            "labels": torch.tensor([0], dtype=torch.int64),
        }]


class FakeResidualCNN(torch.nn.Module):
    def forward(self, tensor):
        assert not torch.is_grad_enabled()
        return tensor


class FakeEnhancer:
    def __init__(self):
        self.input_shape = None

    def process(self, image):
        self.input_shape = image.shape
        msr = np.clip(image.astype(np.int16) + 12, 0, 255).astype(np.uint8)
        return {"msr": msr}


def test_process_camera_frame_uses_enhancement_and_detection_pipeline():
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    model = FakeModel()
    enhancer = FakeEnhancer()
    bundle = {
        "model": model,
        "device": "cpu",
        "enhancer": enhancer,
        "classes": ["object"],
        "image_size": 32,
    }

    result = process_camera_frame(frame, bundle, conf_thr=0.1, nms_thr=0.5)

    assert result["original_bgr"].shape == frame.shape
    assert enhancer.input_shape == (48, 64, 3)
    assert result["enhanced_bgr"].shape == (48, 64, 3)
    assert np.any(result["enhanced_bgr"] != 0)
    assert model.calls == 1
    assert len(result["boxes"]) == 1
    assert result["scores"][0] > 0.8
    assert result["labels"][0] == 0


def test_camera_resize_preserves_aspect_ratio_and_caps_resolution():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    resized = resize_camera_frame(frame, max_dimension=640)

    assert resized.shape == (360, 640, 3)


def test_rtc_configuration_uses_yaml_ice_and_optional_server_turn_credentials():
    config = {"ice_servers": [{"urls": ["stun:example.invalid:3478"]}]}
    env = {
        "UIEDR_TURN_URL": "turn:turn.example.invalid:3478,turns:turn.example.invalid:5349",
        "UIEDR_TURN_USERNAME": "server-user",
        "UIEDR_TURN_CREDENTIAL": "server-secret",
    }

    rtc_config = build_rtc_configuration(config, env)

    assert rtc_config["iceServers"][0] == config["ice_servers"][0]
    assert rtc_config["iceServers"][1] == {
        "urls": ["turn:turn.example.invalid:3478", "turns:turn.example.invalid:5349"],
        "username": "server-user",
        "credential": "server-secret",
    }


def test_front_and_back_camera_choices_request_distinct_browser_devices():
    front = camera_media_stream_constraints("Front Camera")
    back = camera_media_stream_constraints("Back Camera")

    assert front["video"]["facingMode"] == {"ideal": "user"}
    assert back["video"]["facingMode"] == {"ideal": "environment"}
    assert front["video"]["facingMode"] != back["video"]["facingMode"]
    assert front["audio"] is back["audio"] is False


def test_live_stats_reset_between_camera_sessions():
    stats = LiveCameraStats()
    rendered = np.zeros((8, 8, 3), dtype=np.uint8)
    stats.update(10.0, 100.0, 2, "cpu", rendered)
    stats.update(11.0, 120.0, 1, "cpu", rendered)

    stats.begin_session()
    values = stats.snapshot()

    assert values["processed_frames"] == 0
    assert values["received_frames"] == 0
    assert values["fps"] == 0.0
    assert values["processing_ms"] == 0.0
    assert values["latest_rendered"] is None
    assert values["status"] == "connecting"


def test_video_callback_processes_repeated_frames_and_measures_fps():
    model = FakeModel()
    bundle = {
        "model": model,
        "device": "cpu",
        "enhancer": FakeEnhancer(),
        "classes": ["object"],
        "image_size": 32,
    }
    stats = LiveCameraStats()
    load_calls = 0
    load_finished = threading.Event()

    def load_bundle():
        nonlocal load_calls
        load_calls += 1
        load_finished.set()
        return bundle

    callback = make_video_frame_callback(load_bundle, 0.1, 0.5, stats)
    frame = np.zeros((32, 32, 3), dtype=np.uint8)

    first = callback(av.VideoFrame.from_ndarray(frame, format="bgr24"))
    assert load_finished.wait(timeout=5)
    for _ in range(100):
        second = callback(av.VideoFrame.from_ndarray(frame, format="bgr24"))
        if model.calls >= 2:
            break
    values = stats.snapshot()

    assert first.width == second.width == 32
    assert first.height == second.height == 32
    assert model.calls == 2
    assert load_calls == 1
    assert values["fps"] > 0
    assert values["processing_ms"] > 0
    assert values["detections"] == 1
    assert values["device"] == "cpu"
    assert values["error"] is None
    assert values["received_frames"] >= 2
    assert values["latest_rendered"].shape == (32, 32, 3)
    assert values["status"] == "streaming"
