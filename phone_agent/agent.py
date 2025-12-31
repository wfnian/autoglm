"""Main PhoneAgent class for orchestrating phone automation."""

import json
import traceback
from dataclasses import dataclass
from typing import Any, Callable

from phone_agent.actions import ActionHandler
from phone_agent.actions.handler import do, finish, parse_action
from phone_agent.config import get_messages, get_system_prompt
from phone_agent.device_factory import get_device_factory
from phone_agent.model import ModelClient, ModelConfig
from phone_agent.model.client import MessageBuilder


import uiautomator2 as u2
import xml.etree.ElementTree as ET
import re
import time
import subprocess


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

def parse_bounds(bounds_str):
    """解析元素边界坐标"""
    pattern = r'\[(\d+),(\d+)\]\[(\d+),(\d+)\]'
    match = re.match(pattern, bounds_str)
    if match:
        x1, y1, x2, y2 = map(int, match.groups())
        return x1, y1, x2, y2
    return None


def center_of_box(box):
    """(x1,y1,x2,y2) -> (cx,cy)"""
    x1, y1, x2, y2 = box
    return (x1 + x2) // 2, (y1 + y2) // 2


def send_email():
    """硬编码的打开行家App并进入发邮件界面的流程"""
    try:
        # 设备连接
        d = u2.connect()
        # 打开行家app
        d.app_start('com.boc.tesip', '.SplashActivity')

        # 等待首页“内网邮箱”元素出现
        # 最多等10秒，直到出现
        d(text="内网邮箱").wait(timeout=10)
        # 获取XML内容
        xml_content = d.dump_hierarchy()
        if xml_content is False:
            raise RuntimeError("无法获取UI XML")

        # 4. 解析XML找到目标节点
        root = ET.fromstring(xml_content)
        target_node = None
        for node in root.iter("node"):
            if node.get("text") == "内网邮箱":
                parent = node.find("..") if node.find("..") is not None else node
                if parent.get("clickable") == "true":
                    target_node = parent
                    break
                else:
                    target_node = node
                    break
        # 有可能行家后台运行但不是在首页，无法获取到内网邮箱节点，则考虑重启app
        if target_node is None:
            print("未找到『内网邮箱』节点，尝试重新启动App...")
            # 冷启动（先 force-stop 再 start）
            d.app_start('com.boc.tesip', '.SplashActivity', stop=True)
            # 等待首页"内网邮箱"元素出现
            d(text="内网邮箱").wait(timeout=10)
            # 重新获取XML内容
            xml_content = d.dump_hierarchy()
            if xml_content is False:
                raise RuntimeError("无法获取UI XML")
            # 重新解析XML找到目标节点
            root = ET.fromstring(xml_content)
            for node in root.iter("node"):
                if node.get("text") == "内网邮箱":
                    parent = node.find("..") if node.find("..") is not None else node
                    if parent.get("clickable") == "true":
                        target_node = parent
                        break
                    else:
                        target_node = node
                        break
            if target_node is None:
                raise RuntimeError("重启后仍未找到『内网邮箱』节点")

        # 5. 解析 bounds 并点击
        bounds = parse_bounds(target_node.get("bounds"))
        if not bounds:
            raise RuntimeError("节点没有 bounds 属性")
        x, y = center_of_box(bounds)
        d.click(x, y)

        # 6. 等待收件箱出现并点击右下角编辑按钮
        d(text="收件箱").wait(timeout=15)
        time.sleep(3)

        width, height = d.window_size()
        x = int(width * 0.85)   # 右侧85%位置
        y = int(height * 0.95)  # 底部95%位置
        d.click(x, y)
        # time.sleep(0.5)

        print("已进入行家发送邮件界面！")
        return True
    except Exception as e:
        print(f"send_email() 执行失败: {e}")
        traceback.print_exc()
        return False


class PhoneAgent:
    """
    AI-powered agent for automating Android phone interactions.
    新增：对特定高频任务（如行家发邮件）提供硬编码快速路径。
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

    def _try_special_task(self, task: str) -> StepResult | None:
        """
        检查是否需要执行硬编码的特殊任务。
        返回 StepResult 表示已处理并结束任务；返回 None 表示不匹配，继续走大模型逻辑。
        """
        keywords = ["发邮件", "发送邮件", "写邮件", "写一封邮件"]
        if any(kw in task for kw in keywords):
            if self.agent_config.verbose:
                print("检测到行家发邮件相关任务，尝试执行硬编码流程...")

            success = send_email()
            if success:
                if self.agent_config.verbose:
                    print("硬编码发邮件流程执行成功，后续仍可交给大模型继续操作（如填写收件人、主题、正文等）。")
                # 这里选择不直接 finished，让大模型继续处理后续填写邮件内容等步骤
                # 如果希望直接结束任务，可改为 finished=True
                return StepResult(
                    success=True,
                    finished=False,  # 改为 False 让大模型继续
                    action=None,
                    thinking="已通过硬编码方式打开行家并进入发邮件界面",
                    message="已进入行家发送邮件界面，可继续指示填写内容并发送。"
                )
            else:
                if self.agent_config.verbose:
                    print("硬编码发邮件流程失败，回落至大模型智能决策。")
                # 失败时回落，不返回 StepResult，继续正常流程
        return None

    def run(self, task: str) -> str:
        self._context = []
        self._step_count = 0

        # 首先尝试特殊硬编码路径
        special_result = self._try_special_task(task)
        if special_result:
            # 如果硬编码成功且我们希望直接结束（根据需求可调整）
            # 这里保持 finished=False，让后续还能继续让模型填写邮件
            if special_result.finished:
                return special_result.message or "Task completed"
            else:
                # 将特殊步骤的结果加入上下文，继续正常循环
                self._context.append(
                    MessageBuilder.create_assistant_message(
                        f"<think>{special_result.thinking}</think><answer>{special_result.message}</answer>"
                    )
                )

        # 第一步（带用户任务）
        result = self._execute_step(task, is_first=True)

        if result.finished:
            return result.message or "Task completed"

        while self._step_count < self.agent_config.max_steps:
            result = self._execute_step(is_first=False)

            if result.finished:
                return result.message or "Task completed"

        return "Max steps reached"

    def step(self, task: str | None = None) -> StepResult:
        is_first = len(self._context) == 0
        if is_first and not task:
            raise ValueError("Task is required for the first step")
        return self._execute_step(task, is_first)

    def reset(self) -> None:
        self._context = []
        self._step_count = 0

    def _execute_step(
        self, user_prompt: str | None = None, is_first: bool = False
    ) -> StepResult:
        self._step_count += 1

        device_factory = get_device_factory()
        screenshot = device_factory.get_screenshot(self.agent_config.device_id)
        ui_xml = device_factory.get_ui_xml(self.agent_config.device_id)
        current_app = device_factory.get_current_app(self.agent_config.device_id)

        if is_first:
            self._context.append(
                MessageBuilder.create_system_message(self.agent_config.system_prompt)
            )

            screen_info = MessageBuilder.build_screen_info(current_app)
            text_content = f"{user_prompt}\n\n{screen_info}"

            self._context.append(
                MessageBuilder.create_user_message_by_xml(
                    text=text_content, xml_content=ui_xml
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

        try:
            action = parse_action(response.action)
        except ValueError:
            if self.agent_config.verbose:
                traceback.print_exc()
            action = finish(message=response.action)

        if self.agent_config.verbose:
            print("-" * 50)
            print(f"🎯 {msgs['action']}:")
            print(json.dumps(action, ensure_ascii=False, indent=2))
            print("=" * 50 + "\n")

        self._context[-1] = MessageBuilder.remove_images_from_message(self._context[-1])

        try:
            result = self.action_handler.execute(
                action, screenshot.width, screenshot.height
            )
        except Exception as e:
            if self.agent_config.verbose:
                traceback.print_exc()
            result = self.action_handler.execute(
                finish(message=str(e)), screenshot.width, screenshot.height
            )

        self._context.append(
            MessageBuilder.create_assistant_message(
                f"<think>{response.thinking}</think><answer>{response.action}</answer>"
            )
        )

        finished = action.get("_metadata") == "finish" or result.should_finish

        if finished and self.agent_config.verbose:
            msgs = get_messages(self.agent_config.lang)
            print("\n" + "🎉 " + "=" * 48)
            print(
                f"✅ {msgs['task_completed']}: {result.message or action.get('message', msgs['done'])}"
            )
            print("=" * 50 + "\n")

        return StepResult(
            success=result.success,
            finished=finished,
            action=action,
            thinking=response.thinking,
            message=result.message or action.get("message"),
        )

    @property
    def context(self) -> list[dict[str, Any]]:
        return self._context.copy()

    @property
    def step_count(self) -> int:
        return self._step_count