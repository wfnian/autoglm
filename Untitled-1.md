@/api_main.py @/phone_agent/agent.py @/phone_agent/model/client.py @/main.py 
我把原来的main.py精简并改成了接口调用形式，如api_main.py所示，效果就是只能调用相关的函数执行在命令行输出，我希望改造成流式的，流式输出希望一个字一个字蹦出来
函数调用逻辑是agent.run_stream调用_execute_step_stream,_execute_step_stream调用self.model_client.request_stream
请求长这样：


基本功能都是实现的，现在有一个问题，
下面的请求发出去之后，等待一段时间，然后所有的东西是一次性瞬间出来的，并不是一个字一个字出来的。这个问题该如何解决？
是异步阻塞或者缓冲区的问题吗？

PS D:\Desktops\code_\Open-AutoGLM_api2> curl -X POST http://127.0.0.1:8006/v1/phoneagent/completions/stream `
>> -H "Content-Type: application/json" `
>> -H "Authorization: Bearer 65b625de556d77b73690ce7065f836ed.3v0ZSPVHf4owCI7z" `
>> -d '{
>>   "model": "autoglm-phone",
>>   "messages": [{"role": "user", "content": "点击打卡就可以完成任务"}],
>>   "temperature": 0.7,
>>   "device_type": "adb"
>> }'
data: {"flag": "text", "content": "用"}
data: {"flag": "text", "content": "户"}
data: {"flag": "text", "content": "要"}
data: {"flag": "text", "content": "求"}
data: {"flag": "text", "content": "点"}
data: {"flag": "text", "content": "击"}
data: {"flag": "text", "content": "\""}
data: {"flag": "text", "content": "打"}
data: {"flag": "text", "content": "卡"}
data: {"flag": "text", "content": "\""}
...
data: {"flag": "text", "content": "完"}
data: {"flag": "text", "content": "成"}
data: {"flag": "text", "content": "。"}
data: {"flag": "finished", "content": "打卡任务已成功完成！我已经点击了\"签到/签退\"按钮，系统已经完成打卡操作。可以看到\"考勤打卡\"卡片现在显示上次打卡时间为15:24（已更新）。"}