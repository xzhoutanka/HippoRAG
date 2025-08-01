"""
LLM适配器，支持多种外部大语言模型
"""
import os
import openai
import anthropic
import google.generativeai as genai
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class LLMConfig:
    """LLM配置类"""
    model_name: str
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout: int = 60

class LLMAdapter:
    """支持多种外部LLM的适配器"""
    
    SUPPORTED_MODELS = {
        'gpt-4o': {'provider': 'openai', 'api_key_env': 'OPENAI_API_KEY'},
        'gpt-4o-mini': {'provider': 'openai', 'api_key_env': 'OPENAI_API_KEY'}, 
        'gpt-4.5': {'provider': 'openai', 'api_key_env': 'OPENAI_API_KEY'},
        'gpt-O3': {'provider': 'openai', 'api_key_env': 'OPENAI_API_KEY'},
        'gpt-O3-mini': {'provider': 'openai', 'api_key_env': 'OPENAI_API_KEY'},
        'gpt-O4-mini': {'provider': 'openai', 'api_key_env': 'OPENAI_API_KEY'},
        'claude-3.5-sonnet': {'provider': 'anthropic', 'api_key_env': 'ANTHROPIC_API_KEY'},
        'claude-4-sonnet': {'provider': 'anthropic', 'api_key_env': 'ANTHROPIC_API_KEY'},
        'gemini-flash-2.5': {'provider': 'google', 'api_key_env': 'GOOGLE_API_KEY'},
        'gemini-pro-2.5': {'provider': 'google', 'api_key_env': 'GOOGLE_API_KEY'},
    }
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = None
        
        # 检查模型是否支持
        if config.model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"不支持的模型: {config.model_name}")
        
        model_info = self.SUPPORTED_MODELS[config.model_name]
        self.provider = model_info['provider']
        
        # 获取API密钥
        api_key = os.getenv(model_info['api_key_env'])
        if not api_key:
            raise ValueError(f"未找到环境变量 {model_info['api_key_env']}")
        
        # 初始化客户端
        self._init_client(api_key)
        
    def _init_client(self, api_key: str):
        """初始化相应的API客户端"""
        try:
            if self.provider == 'openai':
                self.client = openai.OpenAI(api_key=api_key)
            elif self.provider == 'anthropic':
                self.client = anthropic.Anthropic(api_key=api_key)
            elif self.provider == 'google':
                genai.configure(api_key=api_key)
                self.client = genai.GenerativeModel(self._get_google_model_name())
        except Exception as e:
            logger.error(f"初始化 {self.provider} 客户端失败: {e}")
            raise
    
    def _get_google_model_name(self) -> str:
        """获取Google模型的实际名称"""
        google_model_mapping = {
            'gemini-flash-2.5': 'gemini-2.5-flash',  # Gemini 2.5 Flash (正式版)
            'gemini-pro-2.5': 'gemini-2.5-pro'       # Gemini 2.5 Pro (正式版)
        }
        return google_model_mapping.get(self.config.model_name, self.config.model_name)
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """统一的文本生成接口"""
        try:
            if self.provider == 'openai':
                return self._generate_openai(messages, **kwargs)
            elif self.provider == 'anthropic':
                return self._generate_anthropic(messages, **kwargs)
            elif self.provider == 'google':
                return self._generate_google(messages, **kwargs)
        except Exception as e:
            logger.error(f"生成文本失败: {e}")
            raise
    
    def _generate_openai(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """OpenAI API调用"""
        # OpenAI模型名称映射
        openai_model_mapping = {
            'gpt-4.5': 'gpt-4.5-preview',  # GPT-4.5正式版API名称
        }
        
        model_name = openai_model_mapping.get(self.config.model_name, self.config.model_name)
        
        response = self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=kwargs.get('temperature', self.config.temperature),
            max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
            timeout=kwargs.get('timeout', self.config.timeout)
        )
        return response.choices[0].message.content
    
    def _generate_anthropic(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Anthropic API调用"""
        # 转换消息格式
        system_message = ""
        claude_messages = []
        
        for msg in messages:
            if msg['role'] == 'system':
                system_message = msg['content']
            else:
                claude_messages.append({
                    'role': msg['role'],
                    'content': msg['content']
                })
        
        # Claude模型名称映射
        claude_model_mapping = {
            'claude-3.5-sonnet': 'claude-3-5-sonnet-20241022',
            'claude-4-sonnet': 'claude-sonnet-4-20250514'  # Claude 4 Sonnet正式版
        }
        
        model_name = claude_model_mapping.get(self.config.model_name, self.config.model_name)
        
        response = self.client.messages.create(
            model=model_name,
            system=system_message,
            messages=claude_messages,
            temperature=kwargs.get('temperature', self.config.temperature),
            max_tokens=kwargs.get('max_tokens', self.config.max_tokens)
        )
        return response.content[0].text
    
    def _generate_google(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Google Gemini API调用"""
        # 转换消息格式为Gemini格式
        prompt_parts = []
        for msg in messages:
            if msg['role'] == 'system':
                prompt_parts.append(f"System: {msg['content']}")
            elif msg['role'] == 'user':
                prompt_parts.append(f"User: {msg['content']}")
            elif msg['role'] == 'assistant':
                prompt_parts.append(f"Assistant: {msg['content']}")
        
        prompt = "\n\n".join(prompt_parts)
        
        # 配置生成参数
        generation_config = genai.types.GenerationConfig(
            temperature=kwargs.get('temperature', self.config.temperature),
            max_output_tokens=kwargs.get('max_tokens', self.config.max_tokens)
        )
        
        response = self.client.generate_content(
            prompt,
            generation_config=generation_config
        )
        return response.text

def create_llm_adapter(model_name: str, **kwargs) -> LLMAdapter:
    """创建LLM适配器的工厂函数"""
    config = LLMConfig(model_name=model_name, **kwargs)
    return LLMAdapter(config)