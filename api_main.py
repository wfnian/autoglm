import argparse
import os
import shutil
import subprocess
import sys
import json
from urllib.parse import urlparse
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi import HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any
from dataclasses import dataclass, field

from contextlib import asynccontextmanager
from openai import OpenAI

from phone_agent import PhoneAgent
from phone_agent.agent import AgentConfig
from phone_agent.config.apps import list_supported_apps
from phone_agent.device_factory import DeviceType, get_device_factory, set_device_type
from phone_agent.model import ModelConfig
from phone_agent.config.i18n import get_message



import time

from dotenv import load_dotenv
import asyncio

load_dotenv(".env")

model_api_config = {
    "base_url": os.getenv("MODEL_API_BASE_URL", "http://localhost:8000/v1"),
    "api_key": os.getenv("MODEL_API_KEY", "EMPTY"),
    "model_name": os.getenv("MODEL_NAME", "autoglm-phone"),
}


class Message(BaseModel):
    role: str
    content: str


class AgentCompletionRequest(BaseModel):
    model: str
    messages: List[Message] 
    temperature: Optional[float] = 0.7
    device_type: str = "adb"
    device_id: Optional[str] = None
    lang: Optional[str] = "cn"
    max_steps: Optional[int] = 100


@dataclass
class ModelConfig:
    """Configuration for the AI model."""

    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    model_name: str = "autoglm-phone-9b"
    max_tokens: int = 3000
    temperature: float = 0.0
    top_p: float = 0.85
    frequency_penalty: float = 0.2
    extra_body: dict[str, Any] = field(default_factory=dict)
    lang: str = "cn"  # Language for UI messages: 'cn' or 'en'


@dataclass
class ModelResponse:
    """Response from the AI model."""

    thinking: str
    action: str
    raw_content: str
    # Performance metrics
    time_to_first_token: float | None = None  # Time to first token (seconds)
    time_to_thinking_end: float | None = None  # Time to thinking end (seconds)
    total_time: float | None = None  # Total inference time (seconds)


class MessageBuilder:
    """Helper class for building conversation messages."""

    @staticmethod
    def create_system_message(content: str) -> dict[str, Any]:
        """Create a system message."""
        return {"role": "system", "content": content}

    @staticmethod
    def create_user_message(text: str, image_base64: str | None = None) -> dict[str, Any]:
        """
        Create a user message with optional image.

        Args:
            text: Text content.
            image_base64: Optional base64-encoded image.

        Returns:
            Message dictionary.
        """
        content = []

        if image_base64:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                }
            )

        content.append({"type": "text", "text": text})

        return {"role": "user", "content": content}

    @staticmethod
    def create_user_message_by_xml(text: str, xml_content: str | None = None) -> dict[str, Any]:
        """
        Create a user message with optional image.

        Args:
            text: Text content.
            xml_content: Optional xml_content.

        Returns:
            Message dictionary.
        """
        content = []

        res = text

        if xml_content:
            # content.append(
            #     {
            #         "type": "text",
            #         "text": {"ui_xml": xml_content},
            #     }
            # )
            res = f"{text}\n\n<ui_xml>\n{xml_content}\n</ui_xml>"

        content.append({"type": "text", "text": res})

        return {"role": "user", "content": content}

    @staticmethod
    def create_assistant_message(content: str) -> dict[str, Any]:
        """Create an assistant message."""
        return {"role": "assistant", "content": content}

    @staticmethod
    def remove_images_from_message(message: dict[str, Any]) -> dict[str, Any]:
        """
        Remove image content from a message to save context space.

        Args:
            message: Message dictionary.

        Returns:
            Message with images removed.
        """
        if isinstance(message.get("content"), list):
            message["content"] = [item for item in message["content"] if item.get("type") == "text"]
        return message

    @staticmethod
    def build_screen_info(current_app: str, **extra_info) -> str:
        """
        Build screen info string for the model.

        Args:
            current_app: Current app name.
            **extra_info: Additional info to include.

        Returns:
            JSON string with screen info.
        """
        info = {"current_app": current_app, **extra_info}
        return json.dumps(info, ensure_ascii=False)


def check_system_requirements(device_type: DeviceType = DeviceType.ADB, wda_url: str = "http://localhost:8100") -> bool:

    print("🔍 Checking system requirements...")
    print("-" * 50)

    all_passed = True

    # Determine tool name and command
    if device_type == DeviceType.IOS:
        tool_name = "libimobiledevice"
        tool_cmd = "idevice_id"
    else:
        tool_name = "ADB" if device_type == DeviceType.ADB else "HDC"
        tool_cmd = "adb" if device_type == DeviceType.ADB else "hdc"

    # Check 1: Tool installed
    print(f"1. Checking {tool_name} installation...", end=" ")
    if shutil.which(tool_cmd) is None:
        print("❌ FAILED")
        print(f"   Error: {tool_name} is not installed or not in PATH.")
        print(f"   Solution: Install {tool_name}:")
        if device_type == DeviceType.ADB:
            print("     - macOS: brew install android-platform-tools")
            print("     - Linux: sudo apt install android-tools-adb")
            print("     - Windows: Download from https://developer.android.com/studio/releases/platform-tools")
        elif device_type == DeviceType.HDC:
            # TODO HDC
            pass
        else:  # IOS
            # TODO IOS
            pass
        all_passed = False
    else:
        # Double check by running version command
        try:
            if device_type == DeviceType.ADB:
                version_cmd = [tool_cmd, "version"]
            elif device_type == DeviceType.HDC:
                version_cmd = [tool_cmd, "-v"]
            else:  # IOS
                version_cmd = [tool_cmd, "-ln"]

            result = subprocess.run(version_cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                version_line = result.stdout.strip().split("\n")[0]
                print(f"✅ OK ({version_line if version_line else 'installed'})")
            else:
                print("❌ FAILED")
                print(f"   Error: {tool_name} command failed to run.")
                all_passed = False
        except FileNotFoundError:
            print("❌ FAILED")
            print(f"   Error: {tool_name} command not found.")
            all_passed = False
        except subprocess.TimeoutExpired:
            print("❌ FAILED")
            print(f"   Error: {tool_name} command timed out.")
            all_passed = False

    # If ADB is not installed, skip remaining checks
    if not all_passed:
        print("-" * 50)
        print("❌ System check failed. Please fix the issues above.")
        return False

    # Check 2: Device connected
    print("2. Checking connected devices...", end=" ")
    try:
        if device_type == DeviceType.ADB:
            result = subprocess.run(["adb", "devices"], capture_output=True, text=True, timeout=10)
            lines = result.stdout.strip().split("\n")
            # Filter out header and empty lines, look for 'device' status
            devices = [line for line in lines[1:] if line.strip() and "\tdevice" in line]
        elif device_type == DeviceType.HDC:
            # TODO HDC
            pass
        else:  # IOS
            # TODO IOS
            pass

        if not devices:
            print("❌ FAILED")
            print("   Error: No devices connected.")
            print("   Solution:")
            if device_type == DeviceType.ADB:
                print("     1. Enable USB debugging on your Android device")
                print("     2. Connect via USB and authorize the connection")
                print("     3. Or connect remotely: python main.py --connect <ip>:<port>")
            elif device_type == DeviceType.HDC:
                print("     1. Enable USB debugging on your HarmonyOS device")
                print("     2. Connect via USB and authorize the connection")
                print("     3. Or connect remotely: python main.py --device-type hdc --connect <ip>:<port>")
            else:  # IOS
                print("     1. Connect your iOS device via USB")
                print("     2. Unlock device and tap 'Trust This Computer'")
                print("     3. Verify: idevice_id -l")
                print("     4. Or connect via WiFi using device IP")
            all_passed = False
        else:
            if device_type == DeviceType.ADB:
                device_ids = [d.split("\t")[0] for d in devices]
            elif device_type == DeviceType.HDC:
                device_ids = [d.strip() for d in devices]
            else:  # IOS
                device_ids = devices
            print(
                f"✅ OK ({len(devices)} device(s): {', '.join(device_ids[:2])}{'...' if len(device_ids) > 2 else ''})"
            )
    except subprocess.TimeoutExpired:
        print("❌ FAILED")
        print(f"   Error: {tool_name} command timed out.")
        all_passed = False
    except Exception as e:
        print("❌ FAILED")
        print(f"   Error: {e}")
        all_passed = False

    # If no device connected, skip ADB Keyboard check
    if not all_passed:
        print("-" * 50)
        print("❌ System check failed. Please fix the issues above.")
        return False

    # Check 3: ADB Keyboard installed (only for ADB) or WebDriverAgent (for iOS)
    if device_type == DeviceType.ADB:
        print("3. Checking ADB Keyboard...", end=" ")
        try:
            result = subprocess.run(
                ["adb", "shell", "ime", "list", "-s"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            ime_list = result.stdout.strip()

            if "com.android.adbkeyboard/.AdbIME" in ime_list:
                print("✅ OK")
            else:
                print("❌ FAILED")
                print("   Error: ADB Keyboard is not installed on the device.")

                all_passed = False
        except subprocess.TimeoutExpired:
            print("❌ FAILED")
            print("   Error: ADB command timed out.")
            all_passed = False
        except Exception as e:
            print("❌ FAILED")
            print(f"   Error: {e}")
            all_passed = False
    elif device_type == DeviceType.HDC:
        # TODO HDC
        pass
    else:  # IOS
        # TODO IOS
        pass

    print("-" * 50)

    if all_passed:
        print("✅ All system checks passed!\n")
    else:
        print("❌ System check failed. Please fix the issues above.")

    return all_passed


def check_model_api(base_url: str, model_name: str, api_key: str = "EMPTY") -> bool:
    """
    Check if the model API is accessible and the specified model exists.

    Checks:
    1. Network connectivity to the API endpoint
    2. Model exists in the available models list

    Args:
        base_url: The API base URL
        model_name: The model name to check
        api_key: The API key for authentication

    Returns:
        True if all checks pass, False otherwise.
    """
    print("🔍 Checking model API...")
    print("-" * 50)

    all_passed = True

    # Check 1: Network connectivity using chat API
    print(f"1. Checking API connectivity ({base_url})...", end=" ")
    try:
        # Create OpenAI client
        client = OpenAI(base_url=base_url, api_key=api_key, timeout=30.0)

        # Use chat completion to test connectivity (more universally supported than /models)
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5,
            temperature=0.0,
            stream=False,
        )

        # Check if we got a valid response
        if response.choices and len(response.choices) > 0:
            print("✅ OK")
        else:
            print("❌ FAILED")
            print("   Error: Received empty response from API")
            all_passed = False

    except Exception as e:
        print("❌ FAILED")
        error_msg = str(e)

        # Provide more specific error messages
        if "Connection refused" in error_msg or "Connection error" in error_msg:
            print(f"   Error: Cannot connect to {base_url}")
            print("   Solution:")
            print("     1. Check if the model server is running")
            print("     2. Verify the base URL is correct")
            print(f"     3. Try: curl {base_url}/chat/completions")
        elif "timed out" in error_msg.lower() or "timeout" in error_msg.lower():
            print(f"   Error: Connection to {base_url} timed out")
            print("   Solution:")
            print("     1. Check your network connection")
            print("     2. Verify the server is responding")
        elif "Name or service not known" in error_msg or "nodename nor servname" in error_msg:
            print(f"   Error: Cannot resolve hostname")
            print("   Solution:")
            print("     1. Check the URL is correct")
            print("     2. Verify DNS settings")
        else:
            print(f"   Error: {error_msg}")

        all_passed = False

    print("-" * 50)

    if all_passed:
        print("✅ Model API checks passed!\n")
    else:
        print("❌ Model API check failed. Please fix the issues above.")

    return all_passed


@asynccontextmanager
async def lifespan(app: FastAPI):
    system_status = await asyncio.to_thread(check_system_requirements)
    if not system_status:
        raise RuntimeError("系统环境检查未通过")

    model_status = await asyncio.to_thread(
        check_model_api,
        base_url=model_api_config["base_url"],
        model_name=model_api_config["model_name"],
        api_key=model_api_config["api_key"],
    )

    if not model_status:
        raise RuntimeError("模型API检查未通过")

    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/hello")
def say_hello(name: str = "World"):
    return {"message": f"Hello, {name}!"}


@app.get("/favicon.ico")
def get_favicon():
    return {"message": "favicon.ico"}


@app.post("/v1/phoneagent/completions/stream")
async def agent_completions_stream(request: AgentCompletionRequest):
    """
    Stream agent completions with real-time thinking output.
    
    Returns Server-Sent Events (SSE) stream with format:
    data: {"flag": "text", "content": "1"}
    data: {"flag": "text", "content": "2"}
    data: {"flag": "finished", "content": "Task completed"}
    """
    if not request.device_id:
        device_factory = get_device_factory()
        devices = device_factory.list_devices()
    
    agent_config = AgentConfig(
        max_steps=request.max_steps,
        device_id=devices[0].device_id,
        lang=request.lang or "cn",
        verbose=False,  # Disable verbose output for streaming
    )
    model_config = ModelConfig(
        base_url=model_api_config["base_url"],
        model_name=model_api_config["model_name"],
        api_key=model_api_config["api_key"],
    )
    agent = PhoneAgent(
        model_config=model_config,
        agent_config=agent_config,
    )
    
    print(f"\033[93m ================= Streaming: {request.messages[0].content} ================= \033[0m")
    
    async def event_generator():
        # Reset agent state for new task
        agent.reset()
        
        # Execute first step with streaming
        try:
            async for chunk in agent.run_stream_async(request.messages[0].content):
                # Format as SSE
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                # Yield control so Starlette can flush the chunk immediately
                await asyncio.sleep(0)

                if chunk.get("flag") == "finished":
                    return
                    
            # Fallback (should not reach here because run_stream_async emits finished)
            yield f'data: {{"flag": "finished", "content": "Max steps reached"}}\n\n'
                    
        except Exception as e:
            yield f'data: {{"flag": "error", "content": "Stream error: {str(e)}"}}\n\n'
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable buffering for nginx
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8006)
