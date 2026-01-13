"""Screenshot utilities for capturing Android device screen."""

import base64
import os
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from io import BytesIO
from typing import Tuple

from PIL import Image


@dataclass
class Screenshot:
    """Represents a captured screenshot."""

    base64_data: str
    width: int
    height: int
    is_sensitive: bool = False


def get_screenshot(device_id: str | None = None, timeout: int = 10) -> Screenshot:
    """
    Capture a screenshot from the connected Android device.

    Args:
        device_id: Optional ADB device ID for multi-device setups.
        timeout: Timeout in seconds for screenshot operations.

    Returns:
        Screenshot object containing base64 data and dimensions.

    Note:
        If the screenshot fails (e.g., on sensitive screens like payment pages),
        a black fallback image is returned with is_sensitive=True.
    """
    temp_path = os.path.join(tempfile.gettempdir(), f"screenshot_{uuid.uuid4()}.png")
    adb_prefix = _get_adb_prefix(device_id)

    try:
        # Execute screenshot command
        result = subprocess.run(
            adb_prefix + ["shell", "screencap", "-p", "/sdcard/tmp.png"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        # Check for screenshot failure (sensitive screen)
        output = result.stdout + result.stderr
        if "Status: -1" in output or "Failed" in output:
            return _create_fallback_screenshot(is_sensitive=True)

        # Pull screenshot to local temp path
        subprocess.run(
            adb_prefix + ["pull", "/sdcard/tmp.png", temp_path],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if not os.path.exists(temp_path):
            return _create_fallback_screenshot(is_sensitive=False)

        # Read and encode image
        img = Image.open(temp_path)
        width, height = img.size

        # buffered = BytesIO()
        # img.save(buffered, format="PNG")
        # base64_data = base64.b64encode(buffered.getvalue()).decode("utf-8")
        base64_data, _, _ = compress_for_llm_best(img)

        # Cleanup
        os.remove(temp_path)

        return Screenshot(base64_data=base64_data, width=width, height=height, is_sensitive=False)

    except Exception as e:
        print(f"Screenshot error: {e}")
        return _create_fallback_screenshot(is_sensitive=False)


def get_ui_xml(device_id: str | None = None, timeout: int = 20) -> str | None:
    """
    Dump current UI hierarchy XML from the connected Android device.

    Args:
        device_id: Optional ADB device ID for multi-device setups.
        timeout: Timeout in seconds for dump operations.

    Returns:
        XML string if successful, otherwise None.

    Note:
        On some sensitive pages, UIAutomator dump may fail or return empty content.
    """
    temp_path = os.path.join(tempfile.gettempdir(), f"ui_{uuid.uuid4()}.xml")
    adb_prefix = _get_adb_prefix(device_id)

    try:
        print("Dumping UI hierarchy...")
        # Dump UI hierarchy on device
        result = subprocess.run(
            adb_prefix + ["shell", "uiautomator", "dump", "/sdcard/ui.xml"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        # output = result.stdout + result.stderr
        # print(f"UI Automator dump output: {output}")
        # 替换原来的 if "ERROR" ... 部分
        if result.returncode != 0:
            print(f"[Dump failed] returncode={result.returncode}, stderr={result.stderr}")
            return None

        # 即使 stderr 有 Exception，只要 returncode == 0 且提示 "dumped to"，就认为成功
        if "dumped to:" not in result.stdout:
            print(f"[Unexpected output] stdout={result.stdout}")
            return None

        # Pull XML to local temp path
        subprocess.run(
            adb_prefix + ["pull", "/sdcard/ui.xml", temp_path],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if not os.path.exists(temp_path):
            print("UI XML file not found after pull.")
            return None

        # Read XML content
        with open(temp_path, "r", encoding="utf-8") as f:
            xml_content = f.read()

        # Cleanup
        os.remove(temp_path)

        # Basic sanity check
        if "<hierarchy" not in xml_content:
            return None

        return xml_content

    except subprocess.TimeoutExpired:
        return None


def _get_adb_prefix(device_id: str | None) -> list:
    """Get ADB command prefix with optional device specifier."""
    if device_id:
        return ["adb", "-s", device_id]
    return ["adb"]


def _create_fallback_screenshot(is_sensitive: bool) -> Screenshot:
    """Create a black fallback image when screenshot fails."""
    default_width, default_height = 1080, 2400

    black_img = Image.new("RGB", (default_width, default_height), color="black")
    buffered = BytesIO()
    black_img.save(buffered, format="PNG")
    base64_data = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return Screenshot(
        base64_data=base64_data,
        width=default_width,
        height=default_height,
        is_sensitive=is_sensitive,
    )

def compress_for_llm_best(img: Image.Image) -> tuple[str, int, str]:
    """
    压缩图片并选择最小的无损格式进行返回。
    """

    results = []

    for fmt in ("WEBP", "PNG"):
        buffer = BytesIO()

        if fmt == "WEBP":
            img.save(buffer, format="WEBP", lossless=True, method=6)
        else:
            img.save(buffer, format="PNG", optimize=True, compress_level=9)

        data = buffer.getvalue()
        results.append((
            base64.b64encode(data).decode("utf-8"),
            len(data),
            fmt
        ))
    result = list(map(lambda x: x[1:], results))
    print(f"\033[32;93m{result}\033[0m")  # [[2, 3], [5, 6]]
    # 选择体积最小的
    return min(results, key=lambda x: x[1])
