"""Tests for the shared hardware setup module."""

from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from client.hardware import HardwareConfig, HardwareStack, setup_hardware


@pytest.fixture
def mock_camera():
    """Mock CameraCapture that returns synthetic frames."""
    with patch("client.hardware.CameraCapture") as MockCam:
        cam_instance = MagicMock()
        cam_instance.get_rectified_frame.return_value = (b"jpeg", np.zeros((768, 768, 3), dtype=np.uint8), None)
        MockCam.return_value = cam_instance
        yield cam_instance


@pytest.fixture
def mock_display():
    """Mock display functions."""
    with patch("client.hardware.get_projector_resolution", return_value=(1280, 720)):
        yield


class TestSetupHardware:
    def test_returns_hardware_stack(self, mock_camera, mock_display):
        config = HardwareConfig(webcam=0, no_audio=True)
        stack = setup_hardware(config)

        assert isinstance(stack, HardwareStack)
        assert stack.camera is mock_camera
        assert stack.overlay_manager is not None
        assert stack.overlay_state is not None
        assert stack.object_tracker is not None
        assert stack.program_runtime is not None
        assert stack.session_store is not None
        assert stack.audio_player is None  # no_audio=True

        stack.stop()

    def test_custom_notify_fn(self, mock_camera, mock_display):
        messages = []
        config = HardwareConfig(webcam=0, no_audio=True)
        stack = setup_hardware(config, notify_fn=lambda m: messages.append(m))

        # The overlay manager should use our notify_fn
        assert stack.overlay_manager.notify_fn is not None

        stack.stop()

    def test_overlay_manager_wired_to_overlay_state(self, mock_camera, mock_display):
        config = HardwareConfig(webcam=0, no_audio=True)
        stack = setup_hardware(config)

        assert stack.overlay_manager.overlay_state is stack.overlay_state
        assert stack.overlay_state._om is stack.overlay_manager

        stack.stop()

    def test_program_runtime_wired_to_tracker(self, mock_camera, mock_display):
        config = HardwareConfig(webcam=0, no_audio=True)
        stack = setup_hardware(config)

        assert stack.program_runtime._object_tracker is stack.object_tracker

        stack.stop()

    def test_table_api_factory_creates_working_api(self, mock_camera, mock_display):
        config = HardwareConfig(webcam=0, no_audio=True)
        stack = setup_hardware(config)

        # The runtime's factory should create a usable TableAPI
        api = stack.program_runtime._api_factory()
        assert api._osm is stack.overlay_state
        assert api._tracker is stack.object_tracker
        assert api._session is stack.session_store

        stack.stop()


class TestHardwareStackStop:
    def test_stop_cleans_up(self, mock_camera, mock_display):
        config = HardwareConfig(webcam=0, no_audio=True)
        stack = setup_hardware(config)

        stack.stop()

        mock_camera.stop.assert_called_once()
