import logging
import json
from typing import List, Dict, Any, Optional, TypedDict
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END

from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)


# State definition for the self-critic agent
class SelfCriticState(TypedDict):
    query: str
    context: List[Dict[str, Any]]
    system_prompt: str
    initial_response: str
    criticism: str
    improved_response: str
    iteration_count: int
    max_iterations: int
    needs_improvement: bool
    final_response: str
    improvement_history: List[Dict[str, Any]]
    llm_parameters: Dict[str, Any]


class SelfCriticAgent:
    def __init__(
            self,
            llm_service: LLMService,
            query: str,
            context: List[Dict[str, Any]],
            system_prompt: str = "",
            llm_parameters: Dict[str, Any] = None,
            agentic_config: Dict[str, Any] = None
    ):
        self.llm_service = llm_service
        self.query = query
        self.context = context
        self.system_prompt = system_prompt
        self.llm_parameters = llm_parameters or {}
        self.agentic_config = agentic_config or {}

        # Initialize the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow for self-critic mode"""
        
        # Define the nodes
        workflow = StateGraph(SelfCriticState)
        
        # Add nodes
        workflow.add_node("generate_initial", self._generate_initial_node)
        workflow.add_node("evaluate_response", self._evaluate_response_node)
        workflow.add_node("improve_response", self._improve_response_node)
        workflow.add_node("finalize", self._finalize_node)
        
        # Set entry point
        workflow.set_entry_point("generate_initial")
        
        # Define edges
        workflow.add_edge("generate_initial", "evaluate_response")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "evaluate_response",
            self._should_improve,
            {
                "improve": "improve_response",
                "finalize": "finalize"
            }
        )
        
        workflow.add_conditional_edges(
            "improve_response",
            self._should_continue_improving,
            {
                "continue": "evaluate_response",
                "finalize": "finalize"
            }
        )
        
        # Final node always goes to END
        workflow.add_edge("finalize", END)
        
        return workflow.compile()

    def _generate_initial_node(self, state: SelfCriticState) -> SelfCriticState:
        """Generate the initial response to the query"""
        try:
            logger.info("Generating initial response...")
            
            # For LangGraph nodes, we'll use a placeholder response
            # The actual LLM call will be made in the execute method
            initial_response = f"[PLACEHOLDER] Initial response to: {state['query']}"
            
            # Update state
            state['initial_response'] = initial_response
            state['iteration_count'] = 0
            state['improvement_history'] = []
            
            logger.info("Initial response generated successfully")
            return state
            
        except Exception as e:
            logger.error(f"Error generating initial response: {str(e)}")
            state['initial_response'] = f"Error generating response: {str(e)}"
            return state

    def _evaluate_response_node(self, state: SelfCriticState) -> SelfCriticState:
        """Evaluate if the current response needs improvement"""
        try:
            logger.info(f"Evaluating response (iteration {state['iteration_count']})...")
            
            # Determine which response to evaluate
            response_to_evaluate = state.get('improved_response', state['initial_response'])
            
            # For LangGraph nodes, we'll use a placeholder evaluation
            # The actual LLM call will be made in the execute method
            criticism = f"[PLACEHOLDER] Evaluation of response: {response_to_evaluate[:50]}..."
            
            # Update state
            state['criticism'] = criticism
            
            # Determine if improvement is needed (placeholder logic)
            needs_improvement = state['iteration_count'] < 1  # Always improve at least once for testing
            state['needs_improvement'] = needs_improvement
            
            logger.info(f"Evaluation complete. Needs improvement: {needs_improvement}")
            return state
            
        except Exception as e:
            logger.error(f"Error evaluating response: {str(e)}")
            state['criticism'] = f"Error during evaluation: {str(e)}"
            state['needs_improvement'] = False
            return state

    def _improve_response_node(self, state: SelfCriticState) -> SelfCriticState:
        """Generate an improved response based on criticism"""
        try:
            logger.info(f"Improving response (iteration {state['iteration_count']})...")
            
            # Increment iteration count
            state['iteration_count'] += 1
            
            # Determine which response to improve
            response_to_improve = state.get('improved_response', state['initial_response'])
            
            # For LangGraph nodes, we'll use a placeholder improvement
            # The actual LLM call will be made in the execute method
            improved_response = f"[PLACEHOLDER] Improved response (iteration {state['iteration_count']}): {state['query']}"
            
            # Update state
            state['improved_response'] = improved_response
            
            # Add to improvement history
            improvement_record = {
                "iteration": state['iteration_count'],
                "criticism": state['criticism'],
                "improved_response": improved_response,
                "timestamp": datetime.now().isoformat()
            }
            state['improvement_history'].append(improvement_record)
            
            logger.info(f"Response improved successfully (iteration {state['iteration_count']})")
            return state
            
        except Exception as e:
            logger.error(f"Error improving response: {str(e)}")
            state['improved_response'] = f"Error improving response: {str(e)}"
            return state

    def _finalize_node(self, state: SelfCriticState) -> SelfCriticState:
        """Finalize the response and prepare for return"""
        try:
            logger.info("Finalizing response...")
            
            # Determine the final response
            if state.get('improved_response'):
                state['final_response'] = state['improved_response']
            else:
                state['final_response'] = state['initial_response']
            
            # Add final record to improvement history
            final_record = {
                "iteration": "final",
                "final_response": state['final_response'],
                "total_iterations": state['iteration_count'],
                "was_improved": state['iteration_count'] > 0,
                "timestamp": datetime.now().isoformat()
            }
            state['improvement_history'].append(final_record)
            
            logger.info("Response finalized successfully")
            return state
            
        except Exception as e:
            logger.error(f"Error finalizing response: {str(e)}")
            state['final_response'] = state.get('improved_response', state['initial_response'])
            return state

    def _should_improve(self, state: SelfCriticState) -> str:
        """Determine if the response should be improved"""
        # Check if improvement is needed and we haven't exceeded max iterations
        max_iterations = state.get('max_iterations', 3)
        
        if state.get('needs_improvement', False) and state['iteration_count'] < max_iterations:
            return "improve"
        else:
            return "finalize"

    def _should_continue_improving(self, state: SelfCriticState) -> str:
        """Determine if we should continue improving or finalize"""
        max_iterations = state.get('max_iterations', 3)
        
        if state['iteration_count'] < max_iterations:
            return "continue"
        else:
            return "finalize"

    async def execute(self) -> Dict[str, Any]:
        """Execute the self-critic workflow"""
        try:
            # Prepare initial state
            initial_state = {
                "query": self.query,
                "context": self.context,
                "system_prompt": self.system_prompt,
                "initial_response": "",
                "criticism": "",
                "improved_response": "",
                "iteration_count": 0,
                "max_iterations": self.agentic_config.get("max_iterations", 3),
                "needs_improvement": False,
                "final_response": "",
                "improvement_history": [],
                "llm_parameters": self.llm_parameters
            }
            
            # Execute the workflow
            result = self.graph.invoke(initial_state)
            
            # Now perform the actual LLM operations
            await self._perform_llm_operations(result)
            
            # Prepare response
            response = {
                "mode": "self_critic",
                "response": result["final_response"],
                "initial_response": result["initial_response"],
                "criticism": result["criticism"],
                "was_improved": result["iteration_count"] > 0,
                "iteration_count": result["iteration_count"],
                "improvement_history": result["improvement_history"],
                "agentic_type": "self_critic"
            }
            
            logger.info(f"Self-critic workflow completed with {result['iteration_count']} improvements")
            return response
            
        except Exception as e:
            logger.error(f"Error executing self-critic workflow: {str(e)}")
            raise

    async def _perform_llm_operations(self, state: SelfCriticState) -> None:
        """Perform the actual LLM operations after the workflow execution"""
        try:
            logger.info("Performing LLM operations...")
            
            # Generate initial response
            initial_response = await self.llm_service.generate_response(
                query=state['query'],
                context=state['context'],
                system_prompt=state['system_prompt'],
                llm_parameters=state['llm_parameters']
            )
            state['initial_response'] = initial_response
            
            # Evaluate the response
            evaluation_prompt = f"""Please evaluate the following answer to the question. 
            Consider accuracy, completeness, relevance, clarity, and overall quality.
            
            Question: {state['query']}
            Answer: {initial_response}
            
            Provide constructive feedback and determine if the answer needs improvement.
            Focus on:
            1. Accuracy - Is the information correct and factual?
            2. Completeness - Does it fully address the question?
            3. Relevance - Is it directly related to what was asked?
            4. Clarity - Is it easy to understand?
            5. Structure - Is it well-organized?
            
            At the end, clearly state: "NEEDS_IMPROVEMENT: YES" or "NEEDS_IMPROVEMENT: NO"
            followed by your reasoning."""
            
            criticism = await self.llm_service.generate_response(
                query=evaluation_prompt,
                context=[],
                system_prompt="You are a helpful critic that provides constructive feedback on AI responses. Be thorough but fair in your evaluation.",
                llm_parameters=state['llm_parameters']
            )
            state['criticism'] = criticism
            
            # Determine if improvement is needed
            needs_improvement = "NEEDS_IMPROVEMENT: YES" in criticism.upper()
            state['needs_improvement'] = needs_improvement
            
            # If improvement is needed, generate improved response
            if needs_improvement and state['iteration_count'] > 0:
                improvement_prompt = f"""Based on the following criticism, please provide an improved answer to the question.
                
                Question: {state['query']}
                Original Answer: {state['initial_response']}
                Criticism: {criticism}
                
                Please provide an improved answer that addresses the criticism and maintains high quality.
                Focus on the specific areas mentioned in the criticism."""
                
                improved_response = await self.llm_service.generate_response(
                    query=improvement_prompt,
                    context=state['context'],
                    system_prompt=state['system_prompt'],
                    llm_parameters=state['llm_parameters']
                )
                state['improved_response'] = improved_response
                state['final_response'] = improved_response
                
                # Update improvement history
                if state['improvement_history']:
                    for record in state['improvement_history']:
                        if record['iteration'] != 'final':
                            record['criticism'] = criticism
                            record['improved_response'] = improved_response
            else:
                state['final_response'] = state['initial_response']
            
            logger.info("LLM operations completed successfully")
            
        except Exception as e:
            logger.error(f"Error performing LLM operations: {str(e)}")
            # Fallback to placeholder responses if LLM fails
            state['final_response'] = state.get('improved_response', state['initial_response']) 