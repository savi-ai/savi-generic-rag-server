import logging
import json
from typing import List, Dict, Any, Optional, TypedDict
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END

from app.services.llm_service import LLMService
from app.services.vector_service import VectorService

logger = logging.getLogger(__name__)


# State definition for the agent
class AgentState(TypedDict):
    query: str
    context: List[Dict[str, Any]]
    system_prompt: str
    thought: str
    action: str
    action_input: str
    observation: str
    response: str
    step_count: int
    max_steps: int
    llm_parameters: Dict[str, Any]
    reasoning_steps: List[Dict[str, Any]]


class ReactAgent:
    def __init__(
            self,
            llm_service: LLMService,
            query: str,
            context: List[Dict[str, Any]],
            system_prompt: str = "",
            llm_parameters: Dict[str, Any] = None,
            agentic_config: Dict[str, Any] = None,
            usecase_id: str = None
    ):
        self.llm_service = llm_service
        self.vector_service = VectorService()
        self.query = query
        self.context = context
        self.system_prompt = system_prompt
        self.llm_parameters = llm_parameters or {}
        self.agentic_config = agentic_config or {}
        self.usecase_id = usecase_id

        # Initialize the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""

        # Define the nodes
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("think", self._think_node)
        workflow.add_node("act", self._act_node)
        workflow.add_node("observe", self._observe_node)
        workflow.add_node("decide", self._decide_node)

        # Set entry point
        workflow.set_entry_point("think")

        # Define edges
        workflow.add_edge("think", "act")
        workflow.add_edge("act", "observe")
        workflow.add_edge("observe", "decide")

        # Add conditional edges
        workflow.add_conditional_edges(
            "decide",
            self._should_continue,
            {
                "continue": "think",
                "end": END
            }
        )

        # Set recursion limit
        config = {"recursion_limit": 10}

        return workflow.compile()

    async def _think_node(self, state: AgentState) -> AgentState:
        """Generate thought and decide on action"""
        try:
            # Create a more intelligent thinking prompt
            thinking_prompt = f"""You are a ReAct agent. Analyze the user query and decide what to do next.

User Query: {state['query']}

Available Context: {self._format_context(state['context'])}

Previous Step: {state.get('thought', 'First step')}
Current Step Number: {state['step_count'] + 1}
Maximum Steps: {state['max_steps']}

Previous Observations: {state.get('observation', 'None')}

Think about what you need to do next. Consider:
1. Do you have enough information from the context and previous searches to answer the question?
2. Is the information sufficient and relevant?
3. Would additional searching help improve the answer quality?

Available actions:
- SEARCH: Search for more specific information if needed
- ANSWER: Provide the final answer if you have sufficient information
- END: End the process if you cannot provide a meaningful answer

Respond with ONLY the action you want to take: SEARCH, ANSWER, or END.

If you choose SEARCH, also provide what specific information to search for.
If you choose ANSWER, provide your reasoning for why you can answer now."""

            # Generate thought using LLM
            response = await self.llm_service.generate_response(
                query=thinking_prompt,
                context=[],
                system_prompt="You are a decision-making agent. Respond concisely.",
                llm_parameters=self.llm_parameters
            )

            # Parse the response more robustly
            response_lower = response.lower().strip()

            # Check if we have enough context to answer
            has_sufficient_context = len(state['context']) > 0 and state['step_count'] > 0

            if "answer" in response_lower or "respond" in response_lower or has_sufficient_context:
                state["action"] = "answer"
                state["action_input"] = ""
                state["thought"] = "Have sufficient information to provide a comprehensive answer"
            elif "search" in response_lower and state['step_count'] < state['max_steps'] - 1:
                state["action"] = "search"
                # Extract search query from response
                if ":" in response:
                    state["action_input"] = response.split(":", 1)[1].strip()
                else:
                    state["action_input"] = state["query"]
                state["thought"] = f"Need to search for more specific information: {state['action_input']}"
            else:
                state["action"] = "end"
                state["action_input"] = ""
                state["thought"] = "No more actions needed or max steps reached"

            logger.info(f"Thought: {state['thought']}")
            logger.info(f"Action: {state['action']}")

        except Exception as e:
            logger.error(f"Error in think node: {str(e)}")
            state["thought"] = f"Error in thinking: {str(e)}"
            state["action"] = "end"
            state["action_input"] = ""

        return state

    async def _act_node(self, state: AgentState) -> AgentState:
        """Execute the decided action"""
        try:
            action = state["action"]
            action_input = state["action_input"]

            if action == "search":
                # Search for additional information using vector service
                state["observation"] = await self._search_additional_info(action_input)
            elif action == "api_call":
                # Make API call
                state["observation"] = await self._make_api_call(action_input)
            elif action == "answer":
                # Generate final answer
                state["observation"] = await self._generate_answer(state)
            else:
                state["observation"] = f"Unknown action: {action}"

            logger.info(f"Action executed: {action}")

        except Exception as e:
            logger.error(f"Error in act node: {str(e)}")
            state["observation"] = f"Error executing action: {str(e)}"

        return state

    async def _observe_node(self, state: AgentState) -> AgentState:
        """Process the observation from the action"""
        try:
            # Update step count
            state["step_count"] += 1

            # Add reasoning step to history
            if "reasoning_steps" not in state:
                state["reasoning_steps"] = []

            state["reasoning_steps"].append({
                "step": state["step_count"],
                "thought": state["thought"],
                "action": state["action"],
                "action_input": state["action_input"],
                "observation": state["observation"]
            })

            # Log the observation
            logger.info(f"Observation: {state['observation']}")

        except Exception as e:
            logger.error(f"Error in observe node: {str(e)}")

        return state

    async def _decide_node(self, state: AgentState) -> AgentState:
        """Decide whether to continue or end"""
        try:
            # Check if we should continue
            should_continue = (
                    state["step_count"] < state["max_steps"] and
                    state["action"] != "answer" and
                    state["action"] != "end"
            )

            if not should_continue:
                # Generate final response if we haven't already
                if not state.get("response"):
                    state["response"] = await self._generate_final_response(state)

            logger.info(f"Decision: {'continue' if should_continue else 'end'}")

        except Exception as e:
            logger.error(f"Error in decide node: {str(e)}")

        return state

    def _should_continue(self, state: AgentState) -> str:
        """Determine if the agent should continue or end"""
        if state["step_count"] >= state["max_steps"]:
            return "end"
        if state["action"] in ["answer", "end"]:
            return "end"
        return "continue"

    async def _search_additional_info(self, search_query: str) -> str:
        """Search for additional information using vector service"""
        try:
            if not self.usecase_id:
                return f"Search results for: {search_query} (no usecase_id provided)"

            # Use vector service to search - corrected method name
            search_results = await self.vector_service.similarity_search(
                usecase_id=self.usecase_id,
                query=search_query,
                top_k=self.agentic_config.get("search_top_k", 3)
            )

            if not search_results:
                # If no search results, provide a helpful message
                return f"No additional information found for: {search_query}. The available context should be sufficient to answer the question."

            # Format search results
            formatted_results = []
            for i, result in enumerate(search_results, 1):
                content = result.get("content", result.get("text", str(result)))
                source = result.get("metadata", {}).get("filename", f"Source {i}")
                score = result.get("score", 0.0)
                formatted_results.append(f"{i}. {content}\n   Source: {source} (Score: {score:.3f})")

            return f"Search results for '{search_query}':\n" + "\n".join(formatted_results)

        except Exception as e:
            logger.error(f"Error searching: {str(e)}")
            # If search fails, suggest using available context
            return f"Search failed: {str(e)}. Using available context to provide the best possible answer."

    async def _make_api_call(self, api_params: str) -> str:
        """Make an API call"""
        try:
            # Parse API parameters
            try:
                params = json.loads(api_params)
            except json.JSONDecodeError:
                params = {"url": api_params}

            # This would integrate with your API calling service
            # For now, return a placeholder
            return f"API call made with params: {params} (placeholder - integrate with API service)"
        except Exception as e:
            return f"Error making API call: {str(e)}"

    async def _generate_answer(self, state: AgentState) -> str:
        """Generate the final answer"""
        try:
            # Create the answer prompt
            answer_prompt = f"""Based on the user query and all available information, provide a comprehensive answer.

User Query: {state['query']}

Context: {self._format_context(state['context'])}

Reasoning Process: {state.get('thought', '')}

Additional Information: {state.get('observation', '')}

Please provide a clear, well-reasoned answer that addresses the user's query completely."""

            # Generate response using LLM
            response = await self.llm_service.generate_response(
                query=answer_prompt,
                context=[],
                system_prompt=self.system_prompt,
                llm_parameters=self.llm_parameters
            )

            return response

        except Exception as e:
            return f"Error generating answer: {str(e)}"

    async def _generate_final_response(self, state: AgentState) -> str:
        """Generate the final response when ending"""
        if state.get("response"):
            return state["response"]

        # If no response was generated, create one
        return await self._generate_answer(state)

    def _format_context(self, context: List[Dict[str, Any]]) -> str:
        """Format context for display"""
        if not context:
            return "No context available"

        formatted = []
        for i, item in enumerate(context, 1):
            if isinstance(item, dict):
                content = item.get("content", item.get("text", str(item)))
                source = item.get("metadata", {}).get("filename", f"Source {i}")
                formatted.append(f"{i}. {content}\n   Source: {source}")
            else:
                formatted.append(f"{i}. {str(item)}")

        return "\n".join(formatted)

    async def execute(self) -> Dict[str, Any]:
        """Execute the ReAct agent workflow"""
        try:
            # Initialize state
            initial_state = AgentState(
                query=self.query,
                context=self.context,
                system_prompt=self.system_prompt,
                thought="",
                action="",
                action_input="",
                observation="",
                response="",
                step_count=0,
                max_steps=self.agentic_config.get("max_steps", 5),
                llm_parameters=self.llm_parameters,
                reasoning_steps=[]
            )

            # Execute the graph
            result = await self.graph.ainvoke(initial_state)

            # Return the result with corrected keys
            return {
                "mode": "react",
                "response": result.get("response", ""),
                "thoughts": result.get("thought", ""),
                "actions_taken": result.get("step_count", 0),
                "agentic_type": "react",
                "final_state": result,
                "reasoning_steps": result.get("reasoning_steps", []),
                "final_answer": result.get("response", "")
            }

        except Exception as e:
            logger.error(f"Error executing ReAct agent: {str(e)}")
            raise
