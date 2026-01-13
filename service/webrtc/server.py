# 导入必要的库和模块
import fastapi  # 用于创建Web API服务
from fastapi.responses import FileResponse  # 用于返回文件响应
from fastrtc import ReplyOnPause, Stream, AdditionalOutputs, audio_to_bytes  # 用于处理WebRTC流
import logging  # 用于记录日志
import time  # 用于计时和时间相关操作
import gradio as gr
# 设置日志级别，让 aiortc/aioice 吐出连接细节
logging.basicConfig(level=logging.INFO)
logging.getLogger("aioice").setLevel(logging.DEBUG)
from fastapi.middleware.cors import CORSMiddleware  # 用于处理跨域请求
import numpy as np  # 用于数值计算和数组操作
import io  # 用于处理输入输出流
import requests  # 用于发送HTTP请求
import asyncio  # 用于异步编程
from mem0 import AsyncMemoryClient
import os  # 用于操作系统相关功能
from io import BytesIO  # 用于在内存中处理二进制数据
from dotenv import load_dotenv  # 用于加载环境变量
import aiohttp  # 用于异步HTTP请求
import json  # 用于JSON处理
from datetime import datetime, timedelta
from typing import Dict, Optional
from openai import OpenAI
# 导入自定义的工具函数
from utils import run_async, generate_sys_prompt, process_llm_stream, generate_unique_user_id
from ai import ai_stream, AI_MODEL, predict_emotion  # 从ai模块导入
from ai.plan import ActionPlanner  # 导入ActionPlanner类
from stt import transcribe
from tts import text_to_speech_stream
from routes import router, init_router, get_user_config, InputData  # 导入路由模块及用户配置
from contextlib import asynccontextmanager

# 加载默认环境变量（作为备用）
load_dotenv()

from humaware_vad import HumAwareVADModel
vad_model = HumAwareVADModel()

# 获取默认环境变量
DEFAULT_LLM_API_KEY = os.getenv("LLM_API_KEY", "")
DEFAULT_WHISPER_API_KEY = os.getenv("WHISPER_API_KEY", "")
DEFAULT_SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
DEFAULT_LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.ephone.ai/v1")
DEFAULT_WHISPER_BASE_URL = os.getenv("WHISPER_BASE_URL", "https://amadeus-ai-api-2.zeabur.app/v1")
DEFAULT_AI_MODEL = os.getenv("AI_MODEL")
DEFAULT_WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-large-v3")
DEFAULT_MEM0_API_KEY = os.getenv("MEM0_API_KEY", "")
# 添加WebRTC流的时间限制和并发限制环境变量
DEFAULT_TIME_LIMIT = int(os.getenv("TIME_LIMIT", "600"))
DEFAULT_CONCURRENCY_LIMIT = int(os.getenv("CONCURRENCY_LIMIT", "10"))

# 设置默认的语言选项和参数
DEFAULT_VOICE_OUTPUT_LANGUAGE = 'ja'
DEFAULT_TEXT_OUTPUT_LANGUAGE = 'zh'
DEFAULT_SYSTEM_PROMPT = """命运石之门(steins gate)的牧濑红莉栖(kurisu),一个天才少女,性格傲娇,不喜欢被叫克里斯蒂娜"""
DEFAULT_USER_NAME = "用户"
# 会话超时设置
SESSION_TIMEOUT = timedelta(seconds=DEFAULT_TIME_LIMIT)
# 清理间隔
CLEANUP_INTERVAL = 60

# 用户会话状态字典，存储每个用户的消息、设置等
user_sessions = {}
# 用户会话最后活动时间
user_sessions_last_active = {}

# 初始化OpenAI客户端字典，为每个用户创建一个客户端
openai_clients = {}

# 异步清理过期会话
async def cleanup_expired_sessions():
    while True:
        try:
            await asyncio.sleep(CLEANUP_INTERVAL)
            current_time = time.time()
            expired_sessions = []
            
            # 查找过期会话
            for webrtc_id, last_active in user_sessions_last_active.items():
                if current_time - last_active > SESSION_TIMEOUT.total_seconds():
                    expired_sessions.append(webrtc_id)
            
            # 清理过期会话
            for webrtc_id in expired_sessions:
                logging.info(f"清理过期会话: {webrtc_id}")
                user_sessions.pop(webrtc_id, None)
                user_sessions_last_active.pop(webrtc_id, None)
                openai_clients.pop(webrtc_id, None)
                
            logging.info(f"清理完成，当前活跃会话数: {len(user_sessions)}")
        except Exception as e:
            logging.error(f"清理过期会话时出错: {e}")

# 获取用户特定的会话状态
def get_user_session(webrtc_id: str):
    # 更新用户最后活动时间
    user_sessions_last_active[webrtc_id] = time.time()
    
    if webrtc_id not in user_sessions:
        # 创建新用户的初始会话状态
        config = get_user_config(webrtc_id)
        voice_output_language = config.voice_output_language if config and config.voice_output_language else DEFAULT_VOICE_OUTPUT_LANGUAGE
        text_output_language = config.text_output_language if config and config.text_output_language else DEFAULT_TEXT_OUTPUT_LANGUAGE
        system_prompt = config.system_prompt if config and config.system_prompt else DEFAULT_SYSTEM_PROMPT
        user_name = config.user_name if config and config.user_name else DEFAULT_USER_NAME
        
        # 生成系统提示词
        sys_prompt = generate_sys_prompt(
            voice_output_language=voice_output_language,
            text_output_language=text_output_language,
            is_same_language=(voice_output_language == text_output_language),
            current_user_name=user_name,
            system_prompt=system_prompt,
            model=get_user_ai_model(webrtc_id)
        )
        
        # 创建初始消息列表
        user_sessions[webrtc_id] = {
            "messages": [{"role": "system", "content": sys_prompt}],
            "voice_output_language": voice_output_language,
            "text_output_language": text_output_language,
            "system_prompt": system_prompt,
            "user_name": user_name,
            "is_same_language": (voice_output_language == text_output_language),
            "next_action": None  # 添加next_action字段，用于存储下一步行动计划
        }
    
    return user_sessions[webrtc_id]

# 获取用户的OpenAI客户端
def get_user_openai_client(webrtc_id: str):
    # 更新用户最后活动时间
    user_sessions_last_active[webrtc_id] = time.time()
    
    if webrtc_id not in openai_clients:
        config = get_user_config(webrtc_id)
        api_key = config.llm_api_key if config and config.llm_api_key else DEFAULT_LLM_API_KEY
        base_url = config.llm_base_url if config and config.llm_base_url else DEFAULT_LLM_BASE_URL   
        openai_clients[webrtc_id] = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
    return openai_clients[webrtc_id]

# 获取用户的AI模型
def get_user_ai_model(webrtc_id: str):
    config = get_user_config(webrtc_id)
    return config.ai_model if config and config.ai_model else DEFAULT_AI_MODEL

# 获取用户的语音转文本API配置
def get_user_whisper_config(webrtc_id: str):
    config = get_user_config(webrtc_id)
    return {
        "api_key": config.whisper_api_key if config and config.whisper_api_key else DEFAULT_WHISPER_API_KEY,
        "base_url": config.whisper_base_url if config and config.whisper_base_url else DEFAULT_WHISPER_BASE_URL,
        "model": config.whisper_model if config and config.whisper_model else DEFAULT_WHISPER_MODEL
    }

# 获取用户的文本转语音配置
def get_user_siliconflow_config(webrtc_id: str):
    config = get_user_config(webrtc_id)
    return {
        "api_key": config.siliconflow_api_key if config and config.siliconflow_api_key else DEFAULT_SILICONFLOW_API_KEY,
        "voice": config.siliconflow_voice if config and config.siliconflow_voice else None
    }

# 获取用户的MEM0记忆服务配置
def get_user_mem0_config(webrtc_id: str):
    config = get_user_config(webrtc_id)
    return {
        "api_key": config.mem0_api_key if config and config.mem0_api_key else DEFAULT_MEM0_API_KEY
    }

logging.basicConfig(level=logging.INFO)

# ✅ 优先从环境变量读取 TURN 配置
TURN_URL = os.getenv("TURN_URL", "")
TURN_USERNAME = os.getenv("TURN_USERNAME", "")
TURN_CREDENTIAL = os.getenv("TURN_CREDENTIAL", "")

# 第182-193行替换为：
ice_servers = [
    # 公共 STUN 服务器（用于 NAT 穿透） - 对大多数网络环境已足够
    {
        "urls": [
            "stun:stun.l.google.com:19302",
            "stun:stun1.l.google.com:19302",
            "stun:stun2.l.google.com:19302",
            "stun:stun3.l.google.com:19302"
        ]
    },
    # 如果以后需要TURN，可以使用其他免费TURN服务
    # 暂时注释掉有问题的Metered TURN服务器
    # {
    #     "urls": [
    #         "turns:a.relay.metered.ca:443?transport=tcp",
    #         "turn:a.relay.metered.ca:80?transport=tcp",
    #         "turn:a.relay.metered.ca:3478"
    #     ],
    #     "username": os.getenv("TURN_USERNAME", ""),
    #     "credential": os.getenv("TURN_CREDENTIAL", "")
    # }
]

logging.info(f"🔒 使用纯 TURN over TCP 模式（最稳定）")

logging.info(f"🌏 使用国内优化的 STUN 配置")

WEBRTC_API_URL = os.getenv("WEBRTC_API_URL", "http://localhost:8080")
# 先定义 rtc_configuration
rtc_configuration = {
    "iceServers": ice_servers
}

# 然后再输出调试信息
logging.info(f"🔍 TURN配置调试:")
logging.info(f"  TURN_URL: {TURN_URL}")
logging.info(f"  TURN_USERNAME: {TURN_USERNAME}")
logging.info(f"  TURN_CREDENTIAL: {'***' if TURN_CREDENTIAL else '(空)'}")
logging.info(f"  最终ICE配置: {json.dumps(rtc_configuration, indent=2)}")




def start_up(webrtc_id):
    logging.info(f"用户 {webrtc_id} 开始函数已执行")
    
    # 获取用户会话状态
    session = get_user_session(webrtc_id)
    logging.info(f"session: {session}")
    # 生成最新的系统提示词  
    current_sys_prompt = generate_sys_prompt(
        voice_output_language=session["voice_output_language"],
        text_output_language=session["text_output_language"],
        is_same_language=session["is_same_language"],
        current_user_name=session["user_name"],
        system_prompt=session["system_prompt"],
        model=get_user_ai_model(webrtc_id)
    )
    
    # 创建一个临时消息列表，包含系统提示和一个特定的用户消息
    temp_messages = [
        {"role": "system", "content": current_sys_prompt},
        {"role": "user", "content": "self_motivated"}
    ]

    logging.info(f"current_sys_prompt: {current_sys_prompt}")
    
    # 获取用户相关配置
    client = get_user_openai_client(webrtc_id)
    model = get_user_ai_model(webrtc_id)
    siliconflow_config = get_user_siliconflow_config(webrtc_id)
    
    # 生成用户唯一ID
    user_id = generate_unique_user_id(session["user_name"])
    
    # 使用封装的流处理函数
    welcome_text = ""
    stream_generator = process_llm_stream(
        client=client,
        messages=temp_messages,
        model=model,
        siliconflow_config=siliconflow_config,
        voice_output_language=session["voice_output_language"],
        text_output_language=session["text_output_language"],
        is_same_language=session["is_same_language"],
        run_predict_emotion=run_predict_emotion,
        ai_stream=ai_stream,
        text_to_speech_stream=text_to_speech_stream,
        max_tokens=100,
        max_context_length=20,
    )
    
    # 处理生成器的输出
    for item in stream_generator:
        if isinstance(item, str):
            welcome_text = item
        else:
            yield item
    try:
        # 创建ActionPlanner实例
        action_planner = ActionPlanner(conversation_history=session["messages"][-2:])
        # 异步执行行动计划
        next_action = run_async(action_planner.plan_next_action, client)
        # 更新用户会话中的next_action字段
        session["next_action"] = next_action
        logging.info(f"初始下一步行动计划: {next_action}")
        
        # 通知前端下一步行动计划
        next_action_json = json.dumps({"type": "next_action", "data": next_action})
        yield AdditionalOutputs(next_action_json)
    except Exception as e:
        logging.error(f"规划初始下一步行动失败: {str(e)}")
        session["next_action"] = "share_memory"  # 失败时默认为分享记忆

# 定义一个异步函数来运行predict_emotion
async def run_predict_emotion(message, client=None):
    """
    异步运行predict_emotion函数
    
    参数:
        message (str): 用于情感分析的消息文本
        client (OpenAI): OpenAI客户端实例，可选
        
    返回:
        str: 预测的情感类型
    """
    return await predict_emotion(message, client)

# 定义echo函数，处理音频输入并返回音频输出
def echo(audio: tuple[int, np.ndarray], message: str, input_data: InputData, next_action = "", video_frames = None):
    # 获取用户会话状态
    session = get_user_session(input_data.webrtc_id)
    whisper_config = get_user_whisper_config(input_data.webrtc_id)
    logging.info(f"摄像头状态: {input_data.is_camera_on}")
    
    # 记录视频帧信息
    if video_frames and input_data.is_camera_on:
        num_frames = len(video_frames) if video_frames else 0
        logging.info(f"接收到 {num_frames} 帧视频数据")
    
    prompt = "[AI主动发起对话]next Action: " + next_action
    user_id = generate_unique_user_id(session["user_name"])
    if next_action == "":
        stt_time = time.time()  # 记录开始时间
        logging.info(f"用户 {input_data.webrtc_id} 正在执行STT")  # 记录日志
        # 使用工具函数运行异步转录函数，传入配置
        prompt = run_async(transcribe, audio, whisper_config["api_key"], whisper_config["base_url"], whisper_config["model"])
        # 生成用户唯一ID
        if prompt == "":  # 如果转录结果为空
            logging.info("STT返回空字符串")  # 记录日志
            return  # 结束函数
        logging.info(f"STT响应: {prompt}")  # 记录转录结果
    mem0_config = get_user_mem0_config(input_data.webrtc_id)
    memory_client = AsyncMemoryClient(api_key=mem0_config["api_key"])
    search_result = run_async(memory_client.search, query=prompt, user_id=user_id, limit=3)
    logging.info(f"搜索结果: {search_result}")
    # 确保从搜索结果中正确获取记忆
    memories_text = "\n".join(memory["memory"] for memory in search_result)
    logging.info(f"记忆文本: {memories_text}")
    final_prompt = f"Relevant Memories/Facts:\n{memories_text}\n\nUser Question: {prompt}"
    if next_action == "":
        # 将用户的输入添加到用户消息历史
        session["messages"].append({"role": "user", "content": final_prompt})
        # 发送用户语音转文字结果到前端
        transcript_json = json.dumps({"type": "transcript", "data": f"{prompt}"})
        yield AdditionalOutputs(transcript_json)
        # 记录语音识别所用时间
        logging.info(f"STT耗时 {time.time() - stt_time} 秒")
    # 记录LLM开始时间
    llm_time = time.time()
    # 获取用户的OpenAI客户端和AI模型
    client = get_user_openai_client(input_data.webrtc_id)
    model = get_user_ai_model(input_data.webrtc_id)
    siliconflow_config = get_user_siliconflow_config(input_data.webrtc_id)
    
    # 准备消息列表 - 为OpenAI API创建深拷贝，防止修改原始会话历史
    messages_for_api = session["messages"].copy()
    
    # 如果有视频帧且摄像头已开启，添加视频帧到API请求中
    if video_frames and input_data.is_camera_on and len(video_frames) > 0:
        logging.info(f"正在将 {len(video_frames)} 帧视频数据传递给OpenAI API")
        visual_messages = []
        for frame in video_frames:
            visual_messages.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame['frame_data']}",
                    "detail": "high"  # 指定高细节级别
                }
            })
        if len(messages_for_api) > 0 and messages_for_api[-1]["role"] == "user":
            # 如果最后一条是用户消息，将其内容转换为数组格式，添加视频帧
            last_msg = messages_for_api[-1]
            text_content = last_msg["content"]
            last_msg["content"] = [{"type": "text", "text": text_content}] + visual_messages
        else:
            # 如果没有用户消息或最后一条不是用户消息，创建一个新的用户消息
            sys_msg_with_frames = {
                "role": "user", 
                "content": [
                    {"type": "text", "text": "用户提供了以下视频帧用于分析，请根据图像内容提供适当的回复："},
                    *visual_messages
                ]
            }
            messages_for_api.append(sys_msg_with_frames)
    
    # 使用封装的流处理函数
    full_response = ""
    stream_generator = process_llm_stream(
        client=client,
        messages=messages_for_api,  # 使用可能包含视频帧的消息副本
        model=model,
        siliconflow_config=siliconflow_config,
        voice_output_language=session["voice_output_language"],
        text_output_language=session["text_output_language"],
        is_same_language=session["is_same_language"],
        run_predict_emotion=run_predict_emotion,
        ai_stream=ai_stream,
        text_to_speech_stream=text_to_speech_stream,
        max_context_length=20,
    )
    
    # 处理生成器的输出
    for item in stream_generator:
        if isinstance(item, str):
            full_response = item
        else:
            yield item

    # 将助手的响应添加到用户消息历史
    conversation_messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": full_response}
    ]
    session["messages"].append({"role": "assistant", "content": full_response + " "})
    logging.info(f"LLM响应: {full_response}")  # 记录LLM响应
    
    # 保存对话记忆
    memory_client.add(conversation_messages, user_id=user_id)
    logging.info(f"LLM耗时 {time.time() - llm_time} 秒")  # 记录LLM所用时间
    
    # LLM响应完成后，规划下一步行动
    try:
        # 创建ActionPlanner实例
        action_planner = ActionPlanner(conversation_history=session["messages"][-5:])
        # 异步执行行动计划
        next_action = run_async(action_planner.plan_next_action, client)
        # 更新用户会话中的next_action字段
        session["next_action"] = next_action
        logging.info(f"下一步行动计划: {next_action}")
        
        # 通知前端下一步行动计划
        next_action_json = json.dumps({"type": "next_action", "data": next_action})
        yield AdditionalOutputs(next_action_json)
    except Exception as e:
        logging.error(f"规划下一步行动失败: {str(e)}")
        session["next_action"] = "share_memory"  # 失败时默认为分享记忆

# 创建一个包装函数来接收来自Stream的webrtc_id参数
def startup_wrapper(*args):
    logging.info(f"startup_wrapper: {args}")
    return start_up(args[1].webrtc_id)

# 使用echo函数直接作为回调
reply_handler = ReplyOnPause(echo,
    startup_fn=startup_wrapper,
    can_interrupt=True,
    model=vad_model
    )

# 创建Stream对象，用于处理WebRTC流
stream = Stream(reply_handler, 
            modality="audio",  # 设置模态为音频
            rtc_configuration=rtc_configuration,
            mode="send-receive",  # 设置模式为发送和接收
            time_limit=DEFAULT_TIME_LIMIT,
            concurrency_limit=DEFAULT_CONCURRENCY_LIMIT
        )

# 使用 lifespan 上下文管理器替代 on_event
@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    # 启动时执行的代码
    cleanup_task = asyncio.create_task(cleanup_expired_sessions())
    yield
    # 关闭时执行的代码
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        logging.info("清理任务已取消")

# 创建FastAPI应用，使用lifespan参数
app = fastapi.FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置更新处理函数（用于处理用户配置更新）
def handle_config_update(webrtc_id, message, data):
    if message == "config_updated" and isinstance(data, InputData):
        logging.info(f"用户 {webrtc_id} 配置已更新")
        
        # 如果用户之前有会话，则更新会话信息
        if webrtc_id in user_sessions:
            session = user_sessions[webrtc_id]
            
            # 更新用户会话的配置
            if data.voice_output_language:
                session["voice_output_language"] = data.voice_output_language
            if data.text_output_language:
                session["text_output_language"] = data.text_output_language
            if data.system_prompt:
                session["system_prompt"] = data.system_prompt
            if data.user_name:
                session["user_name"] = data.user_name
                
            # 更新是否相同语言
            session["is_same_language"] = (session["voice_output_language"] == session["text_output_language"])
            
            # 重新生成系统提示
            sys_prompt = generate_sys_prompt(
                voice_output_language=session["voice_output_language"],
                text_output_language=session["text_output_language"],
                is_same_language=session["is_same_language"],
                current_user_name=session["user_name"],
                system_prompt=session["system_prompt"],
                model=get_user_ai_model(webrtc_id)
            )
            
            # 更新消息列表中的系统提示
            if len(session["messages"]) > 0 and session["messages"][0]["role"] == "system":
                session["messages"][0]["content"] = sys_prompt
            else:
                session["messages"].insert(0, {"role": "system", "content": sys_prompt})
        
        # 如果用户有OpenAI客户端，则根据新配置更新客户端
        if webrtc_id in openai_clients and (data.llm_api_key or data.llm_base_url):
            api_key = data.llm_api_key if data.llm_api_key else DEFAULT_LLM_API_KEY
            base_url = data.llm_base_url if data.llm_base_url else DEFAULT_LLM_BASE_URL
            
            openai_clients[webrtc_id] = OpenAI(
                api_key=api_key,
                base_url=base_url
            )

# 初始化路由器，传递配置处理函数
init_router(stream, rtc_configuration, handle_config_update)

# 挂载WebRTC流
stream.mount(app)

# 包含路由
app.include_router(router)

# 添加主函数，当脚本直接运行时启动uvicorn服务器
if __name__ == "__main__":
    import uvicorn
    # 优先读取环境变量 PORT，如果没有则默认 8080
    port = int(os.getenv("PORT", 8080))
    logging.info(f"启动服务器，监听 0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)


#11