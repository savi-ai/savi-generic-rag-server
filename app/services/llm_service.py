import logging
from typing import List, Dict, Any, Optional
import openai
import anthropic
import aiohttp
import json
from datetime import datetime

from app.core.config import settings
import ollama

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
        self.ollama_base_url = settings.ollama_base_url
        self.ollama_model = settings.ollama_model
        self._init_clients()
    
    def _init_clients(self):
        """Initialize LLM clients"""
        if settings.openai_api_key:
            openai.api_key = settings.openai_api_key
            self.openai_client = openai
            logger.info("OpenAI client initialized")
        
        if settings.anthropic_api_key:
            self.anthropic_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            logger.info("Anthropic client initialized")
        
        # Ollama is always available if running locally
        logger.info(f"Ollama client initialized with base URL: {self.ollama_base_url}, model: {self.ollama_model}")
    
    def _format_context(self, search_results: List[Dict[str, Any]]) -> str:
        """Format search results into context for LLM"""
        if not search_results:
            return "No relevant documents found."
        
        context_parts = []
        for i, result in enumerate(search_results, 1):
            content = result.get("content", "")
            metadata = result.get("metadata", {})
            score = result.get("score", 0)

            context_parts.append(f"\n {content}")
            # context_parts.append(f"Document {i} (Relevance: {score:.2f}):")
            if metadata.get("filename"):
                context_parts.append(f"Source: {metadata['filename']}")

            context_parts.append("")
        
        return "\n".join(context_parts)
    
    async def generate_response(
        self,
        query: str,
        context: List[Dict[str, Any]],
        system_prompt: str = "",
        llm_parameters: Dict[str, Any] = None
    ) -> str:
        """Generate response using LLM"""
        try:
            if llm_parameters is None:
                llm_parameters = {}
            
            # Format context
            formatted_context = self._format_context(context)
            
            # Prepare system prompt
            if not system_prompt:
                system_prompt = """You are a helpful AI assistant that answers questions based on the provided context. 
                Always provide accurate and relevant answers based on the information given. 
                If the context doesn't contain enough information to answer the question, say so clearly."""
            
            # Add context to system prompt
            full_system_prompt = f"{system_prompt}\n\nContext:\n{formatted_context}"
            
            # Prepare user message
            user_message = f"Question: {query}"

            # full_message = f"{system_prompt}\n\nContext:\n{formatted_context}\n\n\n{user_message}"
            # Get LLM provider
            provider = llm_parameters.get("provider", settings.default_llm_provider)
            
            if provider == "ollama":
                return await self._generate_ollama_response(
                    system_prompt=full_system_prompt,
                    user_message=user_message,
                    llm_parameters=llm_parameters
                )
            elif provider == "openai" and self.openai_client:
                return await self._generate_openai_response(
                    system_prompt=full_system_prompt,
                    user_message=user_message,
                    llm_parameters=llm_parameters
                )
            elif provider == "anthropic" and self.anthropic_client:
                return await self._generate_anthropic_response(
                    system_prompt=full_system_prompt,
                    user_message=user_message,
                    llm_parameters=llm_parameters
                )
            else:
                raise Exception(f"LLM provider '{provider}' not available or not configured")
                
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            raise
    
    async def _generate_ollama_response(
        self,
        system_prompt: str,
        user_message: str,
        llm_parameters: Dict[str, Any]
    ) -> str:
        """Generate response using Ollama"""
        try:
            temperature = llm_parameters.get("temperature", settings.default_temperature)
            max_tokens = llm_parameters.get("max_tokens", settings.default_max_tokens)

            # print(system_prompt)
            response = ollama.chat(
                model=self.ollama_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                options={"temperature": temperature, "max_tokens": max_tokens}
            )

            return response["message"]["content"]
            
        except Exception as e:
            logger.error(f"Error generating Ollama response: {str(e)}")
            raise
    
    async def _generate_openai_response(
        self,
        system_prompt: str,
        user_message: str,
        llm_parameters: Dict[str, Any]
    ) -> str:
        """Generate response using OpenAI"""
        try:
            model = llm_parameters.get("model", "gpt-3.5-turbo")
            temperature = llm_parameters.get("temperature", settings.default_temperature)
            max_tokens = llm_parameters.get("max_tokens", settings.default_max_tokens)
            
            response = self.openai_client.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating OpenAI response: {str(e)}")
            raise
    
    async def _generate_anthropic_response(
        self,
        system_prompt: str,
        user_message: str,
        llm_parameters: Dict[str, Any]
    ) -> str:
        """Generate response using Anthropic"""
        try:
            model = llm_parameters.get("model", "claude-3-sonnet-20240229")
            temperature = llm_parameters.get("temperature", settings.default_temperature)
            max_tokens = llm_parameters.get("max_tokens", settings.default_max_tokens)
            
            response = self.anthropic_client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_message}
                ]
            )
            
            return response.content[0].text.strip()
            
        except Exception as e:
            logger.error(f"Error generating Anthropic response: {str(e)}")
            raise
    
    async def generate_with_self_critic(
        self,
        query: str,
        context: List[Dict[str, Any]],
        system_prompt: str = "",
        llm_parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Generate response with self-criticism"""
        try:
            # Generate initial response
            initial_response = await self.generate_response(
                query=query,
                context=context,
                system_prompt=system_prompt,
                llm_parameters=llm_parameters
            )
            
            # Generate criticism
            criticism_prompt = f"""Please evaluate the following answer to the question. 
            Consider accuracy, completeness, relevance, and clarity.
            
            Question: {query}
            Answer: {initial_response}
            
            Provide constructive feedback and suggest improvements if needed."""
            
            criticism = await self.generate_response(
                query=criticism_prompt,
                context=[],
                system_prompt="You are a helpful critic that provides constructive feedback on AI responses.",
                llm_parameters=llm_parameters
            )
            
            # Generate improved response if needed
            if "needs improvement" in criticism.lower() or "could be better" in criticism.lower():
                improvement_prompt = f"""Based on the following criticism, please provide an improved answer to the question.
                
                Question: {query}
                Original Answer: {initial_response}
                Criticism: {criticism}
                
                Please provide an improved answer that addresses the criticism."""
                
                improved_response = await self.generate_response(
                    query=improvement_prompt,
                    context=context,
                    system_prompt=system_prompt,
                    llm_parameters=llm_parameters
                )
                
                return {
                    "initial_response": initial_response,
                    "criticism": criticism,
                    "improved_response": improved_response,
                    "was_improved": True
                }
            else:
                return {
                    "initial_response": initial_response,
                    "criticism": criticism,
                    "improved_response": initial_response,
                    "was_improved": False
                }
                
        except Exception as e:
            logger.error(f"Error in self-critic generation: {str(e)}")
            raise 