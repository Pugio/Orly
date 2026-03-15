import threading

import numpy as np
import pytest

from client.session_store import SessionStore


@pytest.fixture
def store(tmp_path):
    return SessionStore(session_dir=str(tmp_path / "session"))


def test_init_creates_directories(store):
    """images_dir and programs_dir exist after init."""
    import os

    assert os.path.isdir(store.images_dir)
    assert os.path.isdir(store.programs_dir)


def test_save_and_load_image(store):
    """Save BGR image, load it back, verify pixel values match."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[10, 20] = [255, 128, 64]
    store.save_image("test-img", img)
    loaded = store.load_image("test-img")
    assert loaded is not None
    np.testing.assert_array_equal(loaded, img)


def test_load_nonexistent_image(store):
    """Returns None for nonexistent image."""
    assert store.load_image("nope") is None


def test_list_images_empty(store):
    """Returns empty list when no images saved."""
    assert store.list_images() == []


def test_list_images(store):
    """Save 3 images, list returns sorted names."""
    for name in ["charlie", "alice", "bob"]:
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        store.save_image(name, img)
    assert store.list_images() == ["alice", "bob", "charlie"]


def test_delete_image(store):
    """Save then delete, returns True, load returns None."""
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    store.save_image("del-me", img)
    assert store.delete_image("del-me") is True
    assert store.load_image("del-me") is None


def test_delete_nonexistent_image(store):
    """Returns False for nonexistent image."""
    assert store.delete_image("nope") is False


def test_save_image_overwrite(store):
    """Save same name twice, second image is what loads."""
    img1 = np.zeros((10, 10, 3), dtype=np.uint8)
    img2 = np.ones((10, 10, 3), dtype=np.uint8) * 200
    store.save_image("overwrite", img1)
    store.save_image("overwrite", img2)
    loaded = store.load_image("overwrite")
    np.testing.assert_array_equal(loaded, img2)


def test_save_and_load_program(store):
    """Save code string, load returns same string."""
    code = "print('hello world')\nx = 42\n"
    store.save_program("my-prog", code)
    assert store.load_program("my-prog") == code


def test_load_nonexistent_program(store):
    """Returns None for nonexistent program."""
    assert store.load_program("nope") is None


def test_list_programs(store):
    """Save 2 programs, list returns sorted names."""
    store.save_program("zebra", "pass")
    store.save_program("alpha", "pass")
    assert store.list_programs() == ["alpha", "zebra"]


def test_get_manifest(store):
    """Verify manifest structure has images, programs, created_at, session_dir."""
    store.save_image("img1", np.zeros((10, 10, 3), dtype=np.uint8))
    store.save_program("prog1", "pass")
    manifest = store.get_manifest()
    assert "images" in manifest
    assert "programs" in manifest
    assert "created_at" in manifest
    assert "session_dir" in manifest
    assert "img1" in manifest["images"]
    assert "prog1" in manifest["programs"]
    assert isinstance(manifest["created_at"], float)


def test_clear(store):
    """Save images and programs, clear, lists are empty."""
    store.save_image("img", np.zeros((10, 10, 3), dtype=np.uint8))
    store.save_program("prog", "pass")
    store.clear()
    assert store.list_images() == []
    assert store.list_programs() == []


def test_sanitize_name_basic():
    """'Hello World' becomes 'hello-world'."""
    assert SessionStore.sanitize_name("Hello World") == "hello-world"


def test_sanitize_name_special_chars():
    """'Scene 1: The Dragon!' becomes 'scene-1-the-dragon'."""
    assert SessionStore.sanitize_name("Scene 1: The Dragon!") == "scene-1-the-dragon"


def test_sanitize_name_unicode():
    """'cafe \u2615' becomes 'caf'."""
    assert SessionStore.sanitize_name("caf\u00e9 \u2615") == "caf"


def test_sanitize_name_empty():
    """Empty string becomes 'unnamed'."""
    assert SessionStore.sanitize_name("") == "unnamed"


def test_sanitize_name_long():
    """200-char string truncated to 100."""
    long_name = "a" * 200
    result = SessionStore.sanitize_name(long_name)
    assert len(result) <= 100


def test_sanitize_name_multiple_hyphens():
    """'a---b' becomes 'a-b'."""
    assert SessionStore.sanitize_name("a---b") == "a-b"


def test_sanitize_name_path_traversal():
    """Path traversal attempts are neutralized."""
    result = SessionStore.sanitize_name("../../../etc/passwd")
    assert ".." not in result
    assert "/" not in result
    assert result == "etc-passwd"


def test_sanitize_name_null_bytes():
    """Null bytes are stripped."""
    result = SessionStore.sanitize_name("hello\x00world")
    assert "\x00" not in result
    assert result == "hello-world"


def test_image_is_png(store):
    """Verify saved file has .png extension and is valid PNG."""
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    path = store.save_image("check-png", img)
    assert path.endswith(".png")
    with open(path, "rb") as f:
        header = f.read(8)
    # PNG magic bytes
    assert header[:4] == b"\x89PNG"


def test_concurrent_saves(store):
    """Save images from 2 threads, both succeed."""
    errors = []

    def save_img(name):
        try:
            img = np.zeros((10, 10, 3), dtype=np.uint8)
            store.save_image(name, img)
        except Exception as e:
            errors.append(e)

    t1 = threading.Thread(target=save_img, args=("thread-1",))
    t2 = threading.Thread(target=save_img, args=("thread-2",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert errors == []
    assert set(store.list_images()) == {"thread-1", "thread-2"}
