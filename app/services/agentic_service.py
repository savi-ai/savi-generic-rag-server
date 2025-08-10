import logging
import json
import aiohttp
from typing import List, Dict, Any, Optional
from datetime import datetime

from app.services.llm_service import LLMService
from app.services.react_agent import ReactAgent
from app.services.self_critic_agent import SelfCriticAgent

logger = logging.getLogger(__name__)

class AgenticService:
    def __init__(self):
        self.llm_service = LLMService()
    
    async def generate_response(
        self,
        query: str,
        context: List[Dict[str, Any]],
        system_prompt: str = "",
        llm_parameters: Dict[str, Any] = None,
        agentic_config: Dict[str, Any] = None,
        usecase_id: str = None
    ) -> Dict[str, Any]:
        """Generate response using agentic mode"""
        try:
            if agentic_config is None:
                agentic_config = {}
            
            agentic_type = agentic_config.get("type", "self_critic")
            
            if agentic_type == "self_critic":
                return await self._self_critic_mode(
                    query=query,
                    context=context,
                    system_prompt=system_prompt,
                    llm_parameters=llm_parameters,
                    agentic_config=agentic_config
                )
            elif agentic_type == "react":
                return await self._react_mode(
                    query=query,
                    context=context,
                    system_prompt=system_prompt,
                    llm_parameters=llm_parameters,
                    agentic_config=agentic_config,
                    usecase_id=usecase_id
                )
            else:
                raise Exception(f"Unknown agentic type: {agentic_type}")
                
        except Exception as e:
            logger.error(f"Error in agentic response generation: {str(e)}")
            raise
    
    async def _self_critic_mode(
        self,
        query: str,
        context: List[Dict[str, Any]],
        system_prompt: str = "",
        llm_parameters: Dict[str, Any] = None,
        agentic_config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Self-critic agentic mode using LangGraph workflow"""
        try:
            # Initialize Self-Critic agent
            self_critic_agent = SelfCriticAgent(
                llm_service=self.llm_service,
                query=query,
                context=context,
                system_prompt=system_prompt,
                llm_parameters=llm_parameters,
                agentic_config=agentic_config
            )
            
            # Execute Self-Critic workflow
            result = await self_critic_agent.execute()
            
            return {
                "mode": "self_critic",
                "response": result["response"],
                "initial_response": result["initial_response"],
                "criticism": result["criticism"],
                "was_improved": result["was_improved"],
                "iteration_count": result["iteration_count"],
                "improvement_history": result["improvement_history"],
                "agentic_type": "self_critic"
            }
            
        except Exception as e:
            logger.error(f"Error in self-critic mode: {str(e)}")
            raise
    
    async def _react_mode(
        self,
        query: str,
        context: List[Dict[str, Any]],
        system_prompt: str = "",
        llm_parameters: Dict[str, Any] = None,
        agentic_config: Dict[str, Any] = None,
        usecase_id: str = None
    ) -> Dict[str, Any]:
        """ReAct (Reasoning and Acting) agentic mode"""
        try:
            # Initialize ReAct agent
            react_agent = ReactAgent(
                llm_service=self.llm_service,
                query=query,
                context=context,
                system_prompt=system_prompt,
                llm_parameters=llm_parameters,
                agentic_config=agentic_config,
                usecase_id=usecase_id
            )
            
            # Execute ReAct reasoning
            result = await react_agent.execute()
            
            return {
                "mode": "react",
                "response": result.get("response", result.get("final_answer", "")),
                "reasoning_steps": result.get("reasoning_steps", []),
                "actions_taken": result.get("actions_taken", 0),
                "agentic_type": "react",
                "thoughts": result.get("thoughts", ""),
                "final_state": result.get("final_state", {})
            }
            
        except Exception as e:
            logger.error(f"Error in ReAct mode: {str(e)}")
            raise 