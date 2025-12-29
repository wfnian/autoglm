from fastapi import FastAPI, Header, HTTPException
from fastapi import Request
from pydantic import BaseModel
from typing import List, Optional
import os
import time

from phone_agent import PhoneAgent
from phone_agent.agent import AgentConfig
from phone_agent.agent_ios import IOSAgentConfig, IOSPhoneAgent
from phone_agent.device_factory import DeviceType, set_device_type
from phone_agent.model import ModelConfig

app = FastAPI()


# ----------- 请求体定义 -------------

class Message(BaseModel):
    role: str
    content: str


class AgentCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7
    device_type: str = "adb"
    device_id: Optional[str] = None
    lang: Optional[str] = "cn"


# ----------- 响应结构 -------------

def build_response(model: str, content: str):
    return {
        "id": f"agent-{int(time.time()*1000)}",
        "object": "agent.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
    }


# ----------- 核心接口 -------------

@app.get("/")
def hello():
    return {"ok": True}



@app.post("/v1/agent/completions")
async def agent_completions(
    request: AgentCompletionRequest,
    authorization: str = Header(default=None),
):

    # --- 校验 API KEY ---
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")

    api_key = authorization.replace("Bearer ", "").strip()

    # --- 获取用户 task（最后一条 user 消息） ---
    user_msgs = [m.content for m in request.messages if m.role == "user"]
    if not user_msgs:
        raise HTTPException(status_code=400, detail="No user message found")
    task = user_msgs[-1]

    # --- 设备类型 ---
    if request.device_type == "adb":
        device_type = DeviceType.ADB
    elif request.device_type == "hdc":
        device_type = DeviceType.HDC
    else:
        device_type = DeviceType.IOS

    if device_type != DeviceType.IOS:
        set_device_type(device_type)

    # --- Model 配置 ---
    base_url = os.getenv("PHONE_AGENT_BASE_URL", "http://localhost:8000/v1")

    model_config = ModelConfig(
        base_url=base_url,
        model_name=request.model,
        api_key=api_key,
        lang=request.lang,
    )

    # --- Agent 配置（按平台区分） ---
    if device_type == DeviceType.IOS:
        agent_config = IOSAgentConfig(
            max_steps=100,
            device_id=request.device_id,
            verbose=True,
            lang=request.lang,
        )
        agent = IOSPhoneAgent(
            model_config=model_config,
            agent_config=agent_config,
        )
    else:
        agent_config = AgentConfig(
            max_steps=100,
            device_id=request.device_id,
            verbose=True,
            lang=request.lang,
        )
        agent = PhoneAgent(
            model_config=model_config,
            agent_config=agent_config,
        )

    # --- 执行动作 ---
    try:
        result = agent.run(task)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # --- 返回 OpenAI 风格 JSON ---
    return build_response(request.model, str(result))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=9966)

"""
python main.py --device-type adb --base-url https://base_url/v1 --model "autoglm-phone" --apikey "your-bigmodel-api-key" "打开美团搜索附近的火锅店"


curl -X POST http://127.0.0.1:9966/v1/agent/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer your-bigmodel-api-key" \
-d '{
  "model": "autoglm-phone",
  "messages": [{"role": "user", "content": "打开美团搜索附近的火锅店"}],
  "max_tokens": 100,
  "temperature": 0.7,
  "device_type": "adb"
}'
"""