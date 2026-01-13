"""Main PhoneAgent class for orchestrating phone automation."""

import json
import traceback
from dataclasses import dataclass
from typing import Any, Callable
import os
import xml.etree.ElementTree as ET

import asyncio

from phone_agent.actions import ActionHandler
from phone_agent.actions.handler import do, finish, parse_action
from phone_agent.config import get_messages, get_system_prompt, get_correct_prompt
from phone_agent.device_factory import get_device_factory
from phone_agent.model import ModelClient, ModelConfig
from phone_agent.model.client import MessageBuilder


@dataclass
class AgentConfig:
    """Configuration for the PhoneAgent."""

    max_steps: int = 100
    device_id: str | None = None
    lang: str = "cn"
    system_prompt: str | None = None
    verbose: bool = True

    def __post_init__(self):
        if self.system_prompt is None:
            self.system_prompt = get_system_prompt(self.lang)


@dataclass
class StepResult:
    """Result of a single agent step."""

    success: bool
    finished: bool
    action: dict[str, Any] | None
    thinking: str
    message: str | None = None


class PhoneAgent:
    """
    AI-powered agent for automating Android phone interactions.

    The agent uses a vision-language model to understand screen content
    and decide on actions to complete user tasks.

    Args:
        model_config: Configuration for the AI model.
        agent_config: Configuration for the agent behavior.
        confirmation_callback: Optional callback for sensitive action confirmation.
        takeover_callback: Optional callback for takeover requests.

    Example:
        >>> from phone_agent import PhoneAgent
        >>> from phone_agent.model import ModelConfig
        >>>
        >>> model_config = ModelConfig(base_url="http://localhost:8000/v1")
        >>> agent = PhoneAgent(model_config)
        >>> agent.run("Open WeChat and send a message to John")
    """

    def __init__(
        self,
        model_config: ModelConfig | None = None,
        agent_config: AgentConfig | None = None,
        confirmation_callback: Callable[[str], bool] | None = None,
        takeover_callback: Callable[[str], None] | None = None,
    ):
        self.model_config = model_config or ModelConfig()
        self.agent_config = agent_config or AgentConfig()

        self.model_client = ModelClient(self.model_config)
        self.action_handler = ActionHandler(
            device_id=self.agent_config.device_id,
            confirmation_callback=confirmation_callback,
            takeover_callback=takeover_callback,
        )

        self._context: list[dict[str, Any]] = []
        self._step_count = 0

    def run(self, task: str) -> str:
        """
        Run the agent to complete a task.

        Args:
            task: Natural language description of the task.

        Returns:
            Final message from the agent.
        """
        self._context = []
        self._step_count = 0

        # First step with user prompt
        result = self._execute_step(task, is_first=True)

        if result.finished:
            return result.message or "Task completed"

        # Continue until finished or max steps reached
        while self._step_count < self.agent_config.max_steps:
            result = self._execute_step(is_first=False)

            if result.finished:
                return result.message or "Task completed"

        return "Max steps reached"

    def run_stream(self, task: str):
        """
        Run the agent to complete a task with streaming response.
        
        Args:
            task: Natural language description of the task.
            
        Yields:
            Streaming chunks from the agent execution.
        """
        self._context = []
        self._step_count = 0

        # First step with user prompt
        for chunk in self._execute_step_stream(task, is_first=True):
            yield chunk
            # Check if we got a finished signal
            if chunk.get("flag") == "finished":
                return

        # Continue until finished or max steps reached
        while self._step_count < self.agent_config.max_steps:
            for chunk in self._execute_step_stream(is_first=False):
                yield chunk
                # Check if we got a finished signal
                if chunk.get("flag") == "finished":
                    return

        # If we reach max steps, yield a finished signal
        yield {"flag": "finished", "content": "Max steps reached"}

    async def run_stream_async(self, task: str):
        """Async streaming version of :meth:`run_stream`.

        This should be used by FastAPI SSE routes so the event loop can flush
        outgoing chunks immediately.
        """
        self._context = []
        self._step_count = 0

        corr_task = await self._correct_asr_text(task)

        async for chunk in self._execute_step_stream_async(corr_task, is_first=True):
            yield chunk
            if chunk.get("flag") == "finished":
                return

        while self._step_count < self.agent_config.max_steps:
            async for chunk in self._execute_step_stream_async(is_first=False):
                yield chunk
                if chunk.get("flag") == "finished":
                    return

        yield {"flag": "finished", "content": "Max steps reached"}

    def step(self, task: str | None = None) -> StepResult:
        """
        Execute a single step of the agent.

        Useful for manual control or debugging.

        Args:
            task: Task description (only needed for first step).

        Returns:
            StepResult with step details.
        """
        is_first = len(self._context) == 0

        if is_first and not task:
            raise ValueError("Task is required for the first step")

        return self._execute_step(task, is_first)

    def reset(self) -> None:
        """Reset the agent state for a new task."""
        self._context = []
        self._step_count = 0

    def _execute_step(self, user_prompt: str | None = None, is_first: bool = False) -> StepResult:
        """Execute a single step of the agent loop."""
        self._step_count += 1

        # Capture current screen state
        device_factory = get_device_factory()
        screenshot = device_factory.get_screenshot(self.agent_config.device_id)
        # ui_xml = device_factory.get_ui_xml(self.agent_config.device_id)
        # print(f"UI XML: {ui_xml}")
        current_app = device_factory.get_current_app(self.agent_config.device_id)
        print(f"UI xml: {ui_xml}")
        # Build messages
        if is_first:
            self._context.append(MessageBuilder.create_system_message(self.agent_config.system_prompt))

            screen_info = MessageBuilder.build_screen_info(current_app)
            text_content = f"{user_prompt}\n\n{screen_info}"
            print(f"\033[41;92m;{text_content}\033[0m")

            # 截图方式1: 原来的方式
            self._context.append(
                MessageBuilder.create_user_message(
                    text=text_content, image_base64=screenshot.base64_data
                )
            )

            # 截图方式2: 包含UI XML
            # print(f"\033[91m{self._context}\033[0m")
            # self._context.append(MessageBuilder.create_user_message_by_xml(text=text_content, xml_content=ui_xml))
        else:
            screen_info = MessageBuilder.build_screen_info(current_app)
            text_content = f"** Screen Info **\n\n{screen_info}"

            self._context.append(
                MessageBuilder.create_user_message(
                    text=text_content, image_base64=screenshot.base64_data
                )
            )
            
            # self._context.append(MessageBuilder.create_user_message_by_xml(text=text_content, xml_content=ui_xml))

        # Get model response
        try:
            msgs = get_messages(self.agent_config.lang)
            print("\n" + "=" * 50)
            print(f"💭 {msgs['thinking']}:")
            print("-" * 50)
            response = self.model_client.request(self._context)
        except Exception as e:
            if self.agent_config.verbose:
                traceback.print_exc()
            return StepResult(
                success=False,
                finished=True,
                action=None,
                thinking="",
                message=f"Model error: {e}",
            )

        # Parse action from response
        try:
            action = parse_action(response.action)
        except ValueError:
            if self.agent_config.verbose:
                traceback.print_exc()
            action = finish(message=response.action)

        if self.agent_config.verbose:
            # Print thinking process
            print("-" * 50)
            print(f"🎯 {msgs['action']}:")
            print(json.dumps(action, ensure_ascii=False, indent=2))
            print("=" * 50 + "\n")

        # Remove image from context to save space
        self._context[-1] = MessageBuilder.remove_images_from_message(self._context[-1])

        # Execute action
        try:
            result = self.action_handler.execute(action, screenshot.width, screenshot.height)
        except Exception as e:
            if self.agent_config.verbose:
                traceback.print_exc()
            result = self.action_handler.execute(finish(message=str(e)), screenshot.width, screenshot.height)

        # Add assistant response to context
        self._context.append(
            MessageBuilder.create_assistant_message(
                # f"<think>{response.thinking}</think><answer>{response.action}</answer>"
                f"{response.thinking}{response.action}"
            )
        )

        # Check if finished
        finished = action.get("_metadata") == "finish" or result.should_finish

        if finished and self.agent_config.verbose:
            msgs = get_messages(self.agent_config.lang)
            print("\n" + "🎉 " + "=" * 48)
            print(f"✅ {msgs['task_completed']}: {result.message or action.get('message', msgs['done'])}")
            print("=" * 50 + "\n")

        return StepResult(
            success=result.success,
            finished=finished,
            action=action,
            thinking=response.thinking,
            message=result.message or action.get("message"),
        )

    def _execute_step_stream(
        self, user_prompt: str | None = None, is_first: bool = False
    ):
        """
        Execute a single step of the agent loop with streaming response.
        
        Yields streaming chunks from model thinking process.
        """
        self._step_count += 1

        # Capture current screen state
        device_factory = get_device_factory()
        screenshot = device_factory.get_screenshot(self.agent_config.device_id)
        current_app = device_factory.get_current_app(self.agent_config.device_id)

        # Build messages
        if is_first:
            self._context.append(MessageBuilder.create_system_message(self.agent_config.system_prompt))

            screen_info = MessageBuilder.build_screen_info(current_app)
            text_content = f"{user_prompt}\n\n{screen_info}"
            
            self._context.append(
                MessageBuilder.create_user_message(
                    text=text_content, image_base64=screenshot.base64_data
                )
            )
        else:
            screen_info = MessageBuilder.build_screen_info(current_app)
            text_content = f"** Screen Info **\n\n{screen_info}"

            self._context.append(
                MessageBuilder.create_user_message(
                    text=text_content, image_base64=screenshot.base64_data
                )
            )

        # Get model streaming response
        try:
            # Call request_stream which yields chunks and ends with a 'response' flag
            response = None
            for chunk in self.model_client.request_stream(self._context):
                if chunk.get("flag") == "response":
                    # This is the ModelResponse
                    response = chunk["content"]
                else:
                    # This is a streaming text chunk
                    yield chunk
                    
            # If we didn't get a response, something went wrong
            if response is None:
                yield {"flag": "error", "content": "Failed to get model response"}
                return
                
        except Exception as e:
            if self.agent_config.verbose:
                traceback.print_exc()
            yield {"flag": "error", "content": f"Model error: {e}"}
            return

        # Parse action from response
        try:
            action = parse_action(response.action)
        except ValueError:
            action = finish(message=response.action)

        # Remove image from context to save space
        self._context[-1] = MessageBuilder.remove_images_from_message(self._context[-1])

        # Execute action
        try:
            result = self.action_handler.execute(action, screenshot.width, screenshot.height)
        except Exception as e:
            result = self.action_handler.execute(finish(message=str(e)), screenshot.width, screenshot.height)

        # Add assistant response to context
        self._context.append(
            MessageBuilder.create_assistant_message(
                f"<think>{response.thinking}</think><answer>{response.action}</answer>"
            )
        )

        # Check if finished
        finished = action.get("_metadata") == "finish" or result.should_finish

        if finished:
            yield {"flag": "finished", "content": result.message or action.get("message", "Task completed")}
            return  # IMPORTANT: Return after sending finished signal

    async def _correct_asr_text(self, asr_text: str) -> str:
        """
        Use the model to correct ASR text based on predefined correction principles.

        Args:
            asr_text: The original ASR text to be corrected.

        Returns:
            Corrected text.
        """
        self._corr_context = []
        self._corr_context.append(
                MessageBuilder.create_system_message(get_correct_prompt())
            ) 
        text_content = f"\n\n以下是语音转文本的待纠正输入：'{asr_text}'"
        self._corr_context.append(
            MessageBuilder.create_user_message(
                text=text_content
            )
        )
        self._model_client = ModelClient(self.model_config)
        response = await self._model_client.async_corr_client.chat.completions.create(
            messages=self._corr_context,
            model=self.model_config.corr_model_name,
            stream=False,
        )
        print(f"\033[91m Corrected ASR Response: {response} \033[0m")
        

    async def _execute_step_stream_async(
        self, user_prompt: str | None = None, is_first: bool = False
    ):
        """Async version of :meth:`_execute_step_stream`.

        Key difference: model streaming is done via AsyncOpenAI and `async for`,
        so FastAPI can flush SSE chunks in real-time.
        """
        self._step_count += 1

        print(f"\033[94m ======= Step {'execute_step'} ======= \033[0m")

        device_factory = get_device_factory()
        screenshot = device_factory.get_screenshot(self.agent_config.device_id)
        current_app = device_factory.get_current_app(self.agent_config.device_id)

        if is_first:
            self._context.append(
                MessageBuilder.create_system_message(self.agent_config.system_prompt)
            )
            screen_info = MessageBuilder.build_screen_info(current_app)
            text_content = f"{user_prompt}\n\n{screen_info}"
            self._context.append(
                MessageBuilder.create_user_message(
                    text=text_content, image_base64=screenshot.base64_data
                )
            )
        else:
            screen_info = MessageBuilder.build_screen_info(current_app)
            text_content = f"** Screen Info **\n\n{screen_info}"
            self._context.append(
                MessageBuilder.create_user_message(
                    text=text_content, image_base64=screenshot.base64_data
                )
            )

        # Get model streaming response (async)
        print(f"\033[94m ======= Step {'model_response'} ======= \033[0m")
        try:
            response = None
            async for chunk in self.model_client.request_stream_async(self._context):
                print(f"\033[94m ======= Chunk: {chunk} ======= \033[0m")
                if chunk.get("flag") == "response":
                    print(f"\033[96m;3m ======= Step {chunk} ======= \033[0m")
                    response = chunk["content"]
                else:
                    yield chunk

            if response is None:
                print(f"\033[94m ======= Step {'error_no_response'} ======= \033[0m")
                yield {"flag": "error", "content": "Failed to get model response"}
                return
        except asyncio.CancelledError:
            print(f"\033[94m ======= Step {'cancelled'} ======= \033[0m")
        except GeneratorExit:
            print(f"\033[94m ======= Step {'generator_exit'} ======= \033[0m")
        except Exception as e:
            print(f"\033[94m ======= Step {'error'} ======= \033[0m")
            if self.agent_config.verbose:
                traceback.print_exc()
            yield {"flag": "error", "content": f"Model error: {e}"}
            return
        print(f"\033[94m ======= Step {'execute_action'} ======= \033[0m")
        # Parse action
        try:
            action = parse_action(response.action)
        except ValueError:
            action = finish(message=response.action)

        # Remove image from context to save space
        self._context[-1] = MessageBuilder.remove_images_from_message(self._context[-1])

        # Execute action (blocking, but happens after thinking stream)
        try:
            result = self.action_handler.execute(action, screenshot.width, screenshot.height)
        except Exception as e:
            result = self.action_handler.execute(
                finish(message=str(e)), screenshot.width, screenshot.height
            )

        self._context.append(
            MessageBuilder.create_assistant_message(
                f"<think>{response.thinking}</think><answer>{response.action}</answer>"
            )
        )

        finished = action.get("_metadata") == "finish" or result.should_finish
        if finished:
            yield {
                "flag": 103,
                "content": result.message or action.get("message", "Task completed"),
            }
            return

        # If not finished, provide a lightweight step marker so clients can
        # distinguish between multi-step cycles (useful for UIs/logging).
        yield {"flag": 102, "content": {"step": self._step_count, "finished": False}}

    @property
    def context(self) -> list[dict[str, Any]]:
        """Get the current conversation context."""
        return self._context.copy()

    @property
    def step_count(self) -> int:
        """Get the current step count."""
        return self._step_count
