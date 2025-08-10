import logging
import time
import asyncio
import os
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime

from app.services.vector_service import VectorService
from app.services.llm_service import LLMService
from app.services.agentic_service import AgenticService
from app.services.guardrail_service import GuardrailService
from app.services.task_manager import task_manager, TaskStatus

logger = logging.getLogger(__name__)

class EvaluationService:
    def __init__(self):
        self.vector_service = VectorService()
        self.llm_service = LLMService()
        self.agentic_service = AgenticService()
        self.guardrail_service = GuardrailService()
    
    def start_async_evaluation(
        self,
        usecase_id: str,
        evaluation_data: List[Dict[str, Any]],
        llm_parameters: Dict[str, Any] = None,
        system_prompt: str = "",
        agentic_config: Dict[str, Any] = None,
        guardrail_config: Dict[str, Any] = None
    ) -> str:
        """Start async evaluation and return task ID"""
        
        # Create task
        task_params = {
            "usecase_id": usecase_id,
            "evaluation_data": evaluation_data,
            "llm_parameters": llm_parameters or {},
            "system_prompt": system_prompt,
            "agentic_config": agentic_config or {},
            "guardrail_config": guardrail_config or {}
        }
        
        task_id = task_manager.create_task("evaluation", task_params)
        
        # Start background task
        asyncio.create_task(self._run_async_evaluation(task_id))
        
        return task_id
    
    async def _run_async_evaluation(self, task_id: str):
        """Run the actual evaluation in background"""
        try:
            task = task_manager.get_task_status(task_id)
            if not task:
                return
            
            params = task["parameters"]
            task_manager.update_task_status(task_id, TaskStatus.RUNNING)
            
            # Run evaluation
            results = await self.evaluate_queries(
                task_id=task_id,
                usecase_id=params["usecase_id"],
                evaluation_data=params["evaluation_data"],
                llm_parameters=params["llm_parameters"],
                system_prompt=params["system_prompt"],
                agentic_config=params["agentic_config"],
                guardrail_config=params["guardrail_config"]
            )
            
            # Generate reports
            await self._generate_reports(task_id, results)
            
            task_manager.update_task_status(task_id, TaskStatus.COMPLETED)
            logger.info(f"Evaluation task {task_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Evaluation task {task_id} failed: {str(e)}")
            task_manager.update_task_status(
                task_id, 
                TaskStatus.FAILED, 
                error_message=str(e)
            )
    
    async def _generate_reports(self, task_id: str, results: Dict[str, Any]):
        """Generate CSV and HTML reports"""
        try:
            results_dir = task_manager.results_dir
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Generate CSV report
            csv_path = os.path.join(results_dir, f"evaluation_{task_id}_{timestamp}.csv")
            await self._generate_csv_report(results, csv_path)
            task_manager.add_result_file(task_id, csv_path, "csv")
            
            # Generate HTML report
            html_path = os.path.join(results_dir, f"evaluation_{task_id}_{timestamp}.html")
            await self._generate_html_report(results, html_path)
            task_manager.add_result_file(task_id, html_path, "html")
            
            logger.info(f"Generated reports for task {task_id}: CSV and HTML")
            
        except Exception as e:
            logger.error(f"Failed to generate reports for task {task_id}: {str(e)}")
            raise
    
    async def _generate_csv_report(self, results: Dict[str, Any], file_path: str):
        """Generate CSV report"""
        try:
            # Flatten the results for CSV
            csv_data = []
            for item in results["results"]:
                row = {
                    "query": item["query"],
                    "expected_answer": item["expected_answer"],
                    "generated_answer": item["generated_answer"],
                    "similarity_score": item["similarity_score"],
                    "response_time_ms": item["response_time_ms"],
                    "status": item["status"],
                    "error": item.get("error", ""),
                    "guardrail_blocked": item.get("guardrail_blocked", False),
                    "guardrail_reason": item.get("guardrail_reason", "")
                }
                csv_data.append(row)
            
            # Create DataFrame and save
            df = pd.DataFrame(csv_data)
            df.to_csv(file_path, index=False)
            
            logger.info(f"CSV report saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate CSV report: {str(e)}")
            raise
    
    async def _generate_html_report(self, results: Dict[str, Any], file_path: str):
        """Generate HTML report"""
        try:
            summary = results["summary"]
            
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .summary {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: white; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .results-table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        .results-table th, .results-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        .results-table th {{ background-color: #f2f2f2; }}
        .status-success {{ background-color: #d4edda; }}
        .status-error {{ background-color: #f8d7da; }}
        .status-blocked {{ background-color: #fff3cd; }}
        .score-high {{ color: #28a745; font-weight: bold; }}
        .score-medium {{ color: #ffc107; font-weight: bold; }}
        .score-low {{ color: #dc3545; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>RAG Evaluation Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p><strong>Use Case:</strong> {results.get('usecase_id', 'N/A')}</p>
    </div>
    
    <div class="summary">
        <h2>Summary</h2>
        <div class="metric">
            <strong>Total Queries:</strong> {summary['total_queries']}
        </div>
        <div class="metric">
            <strong>Successful:</strong> {summary['successful_queries']}
        </div>
        <div class="metric">
            <strong>Failed:</strong> {summary['failed_queries']}
        </div>
        <div class="metric">
            <strong>Blocked by Guardrails:</strong> {summary.get('blocked_queries', 0)}
        </div>
        <div class="metric">
            <strong>Avg Similarity:</strong> {summary['average_similarity']:.3f}
        </div>
        <div class="metric">
            <strong>Avg Response Time:</strong> {summary['average_response_time']:.0f}ms
        </div>
    </div>
    
    <h2>Detailed Results</h2>
    <table class="results-table">
        <thead>
            <tr>
                <th>#</th>
                <th>Query</th>
                <th>Expected Answer</th>
                <th>Generated Answer</th>
                <th>Similarity</th>
                <th>Response Time</th>
                <th>Status</th>
            </tr>
        </thead>
        <tbody>
"""
            
            for i, item in enumerate(results["results"], 1):
                status_class = "status-success" if item["status"] == "success" else ("status-blocked" if item.get("guardrail_blocked") else "status-error")
                
                score = item.get("similarity_score", 0)
                score_class = "score-high" if score > 0.8 else ("score-medium" if score > 0.5 else "score-low")
                
                html_content += f"""
            <tr class="{status_class}">
                <td>{i}</td>
                <td style="max-width: 200px; word-wrap: break-word;">{item['query'][:100]}{'...' if len(item['query']) > 100 else ''}</td>
                <td style="max-width: 200px; word-wrap: break-word;">{item['expected_answer'][:100]}{'...' if len(item['expected_answer']) > 100 else ''}</td>
                <td style="max-width: 200px; word-wrap: break-word;">{item['generated_answer'][:100]}{'...' if len(item['generated_answer']) > 100 else ''}</td>
                <td class="{score_class}">{score:.3f}</td>
                <td>{item['response_time_ms']:.0f}ms</td>
                <td>{item['status']}{' (Guardrail)' if item.get('guardrail_blocked') else ''}</td>
            </tr>
"""
            
            html_content += """
        </tbody>
    </table>
</body>
</html>
"""
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"HTML report saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate HTML report: {str(e)}")
            raise
    
    async def evaluate_queries(
        self,
        task_id: str,
        usecase_id: str,
        evaluation_data: List[Dict[str, Any]],
        llm_parameters: Dict[str, Any] = None,
        system_prompt: str = "",
        agentic_config: Dict[str, Any] = None,
        guardrail_config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Evaluate queries with progress tracking"""
        
        if llm_parameters is None:
            llm_parameters = {}
        if agentic_config is None:
            agentic_config = {}
        if guardrail_config is None:
            guardrail_config = {}
        
        results = []
        total_queries = len(evaluation_data)
        
        # Update total items for progress tracking
        task_manager.update_task_progress(task_id, 0, total_queries)
        
        for i, item in enumerate(evaluation_data):
            try:
                start_time = time.time()
                
                query = item.get("query", "")
                expected_answer = item.get("expected_answer", "")
                
                # Apply question guardrail if configured
                guardrail_blocked = False
                guardrail_reason = ""
                generated_answer = ""
                
                if guardrail_config.get("useGuardrails") and guardrail_config.get("questionGuardrails"):
                    is_allowed, reason = await self.guardrail_service.apply_question_guardrail(
                        query, guardrail_config["questionGuardrails"]
                    )
                    if not is_allowed:
                        guardrail_blocked = True
                        guardrail_reason = f"Question blocked: {reason}"
                        generated_answer = "Sorry could not generate response as guardrails blocked the question"
                
                if not guardrail_blocked:
                    # Perform similarity search
                    search_results = await self.vector_service.similarity_search(
                        usecase_id=usecase_id,
                        query=query,
                        top_k=llm_parameters.get("top_k", 5)
                    )
                    
                    # Check if agentic mode is enabled
                    if agentic_config.get("enabled", False):
                        response_data = await self.agentic_service.generate_response(
                            query=query,
                            context=search_results,
                            system_prompt=system_prompt,
                            llm_parameters=llm_parameters,
                            agentic_config=agentic_config,
                            usecase_id=usecase_id
                        )
                        generated_answer = response_data["response"]
                    else:
                        # Generate response using LLM
                        generated_answer = await self.llm_service.generate_response(
                            query=query,
                            context=search_results,
                            system_prompt=system_prompt,
                            llm_parameters=llm_parameters
                        )
                    
                    # Apply answer guardrail if configured
                    if guardrail_config.get("useGuardrails") and guardrail_config.get("answerGuardrails"):
                        is_allowed, reason = await self.guardrail_service.apply_answer_guardrail(
                            generated_answer, guardrail_config["answerGuardrails"]
                        )
                        if not is_allowed:
                            guardrail_blocked = True
                            guardrail_reason = f"Answer blocked: {reason}"
                            generated_answer = "Sorry could not generate response as guardrails blocked the response"
                
                # Calculate similarity score
                similarity_score = self._calculate_similarity(expected_answer, generated_answer)
                
                end_time = time.time()
                response_time = (end_time - start_time) * 1000  # Convert to milliseconds
                
                result = {
                    "query": query,
                    "expected_answer": expected_answer,
                    "generated_answer": generated_answer,
                    "similarity_score": similarity_score,
                    "response_time_ms": response_time,
                    "status": "success" if not guardrail_blocked else "blocked",
                    "guardrail_blocked": guardrail_blocked,
                    "guardrail_reason": guardrail_reason
                }
                
                results.append(result)
                
                # Update progress
                task_manager.update_task_progress(task_id, i + 1, total_queries)
                
                logger.info(f"Evaluated query {i+1}/{total_queries} for task {task_id}")
                
            except Exception as e:
                logger.error(f"Error evaluating query {i+1}: {str(e)}")
                
                result = {
                    "query": item.get("query", ""),
                    "expected_answer": item.get("expected_answer", ""),
                    "generated_answer": "",
                    "similarity_score": 0.0,
                    "response_time_ms": 0,
                    "status": "error",
                    "error": str(e),
                    "guardrail_blocked": False,
                    "guardrail_reason": ""
                }
                
                results.append(result)
                
                # Update progress even on error
                task_manager.update_task_progress(task_id, i + 1, total_queries)
        
        # Calculate summary statistics
        successful_results = [r for r in results if r["status"] == "success"]
        blocked_results = [r for r in results if r["status"] == "blocked"]
        failed_results = [r for r in results if r["status"] == "error"]
        
        summary = {
            "total_queries": len(results),
            "successful_queries": len(successful_results),
            "blocked_queries": len(blocked_results),
            "failed_queries": len(failed_results),
            "average_similarity": sum(r["similarity_score"] for r in successful_results) / len(successful_results) if successful_results else 0,
            "average_response_time": sum(r["response_time_ms"] for r in successful_results) / len(successful_results) if successful_results else 0
        }
        
        return {
            "usecase_id": usecase_id,
            "results": results,
            "summary": summary,
            "evaluation_completed_at": datetime.now().isoformat()
        }
    
    def _calculate_similarity(self, expected: str, generated: str) -> float:
        """Calculate similarity between expected and generated answers using simple cosine similarity"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            if not expected.strip() or not generated.strip():
                return 0.0
            
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer().fit([expected, generated])
            vectors = vectorizer.transform([expected, generated])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            
            return float(similarity)
            
        except Exception as e:
            logger.warning(f"Error calculating similarity: {str(e)}")
            return 0.0 