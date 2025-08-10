import asyncio
import uuid
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class TaskManager:
    def __init__(self):
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.results_dir = "./evaluation_results"
        os.makedirs(self.results_dir, exist_ok=True)
    
    def create_task(self, task_type: str, parameters: Dict[str, Any]) -> str:
        """Create a new task and return task ID"""
        task_id = str(uuid.uuid4())
        
        self.tasks[task_id] = {
            "id": task_id,
            "type": task_type,
            "status": TaskStatus.PENDING,
            "created_at": datetime.now().isoformat(),
            "started_at": None,
            "completed_at": None,
            "parameters": parameters,
            "progress": 0,
            "total_items": 0,
            "processed_items": 0,
            "error_message": None,
            "result_files": []
        }
        
        logger.info(f"Created task {task_id} of type {task_type}")
        return task_id
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status and details"""
        return self.tasks.get(task_id)
    
    def update_task_status(self, task_id: str, status: TaskStatus, **kwargs):
        """Update task status and additional fields"""
        if task_id in self.tasks:
            self.tasks[task_id]["status"] = status
            
            if status == TaskStatus.RUNNING and "started_at" not in self.tasks[task_id]:
                self.tasks[task_id]["started_at"] = datetime.now().isoformat()
            elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                self.tasks[task_id]["completed_at"] = datetime.now().isoformat()
            
            # Update any additional fields
            for key, value in kwargs.items():
                self.tasks[task_id][key] = value
            
            logger.info(f"Updated task {task_id} status to {status}")
    
    def update_task_progress(self, task_id: str, processed_items: int, total_items: int = None):
        """Update task progress"""
        if task_id in self.tasks:
            self.tasks[task_id]["processed_items"] = processed_items
            if total_items is not None:
                self.tasks[task_id]["total_items"] = total_items
            
            if total_items and total_items > 0:
                self.tasks[task_id]["progress"] = int((processed_items / total_items) * 100)
    
    def add_result_file(self, task_id: str, file_path: str, file_type: str):
        """Add a result file to the task"""
        if task_id in self.tasks:
            self.tasks[task_id]["result_files"].append({
                "path": file_path,
                "type": file_type,
                "created_at": datetime.now().isoformat()
            })
    
    def get_task_result_files(self, task_id: str) -> List[Dict[str, Any]]:
        """Get result files for a task"""
        task = self.tasks.get(task_id)
        if task:
            return task.get("result_files", [])
        return []
    
    def cleanup_old_tasks(self, max_age_hours: int = 24):
        """Clean up old completed/failed tasks"""
        current_time = datetime.now()
        tasks_to_remove = []
        
        for task_id, task in self.tasks.items():
            if task["status"] in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                completed_at = datetime.fromisoformat(task["completed_at"])
                age_hours = (current_time - completed_at).total_seconds() / 3600
                
                if age_hours > max_age_hours:
                    tasks_to_remove.append(task_id)
                    # Also clean up result files
                    for result_file in task.get("result_files", []):
                        try:
                            if os.path.exists(result_file["path"]):
                                os.remove(result_file["path"])
                        except Exception as e:
                            logger.warning(f"Failed to remove file {result_file['path']}: {e}")
        
        for task_id in tasks_to_remove:
            del self.tasks[task_id]
            logger.info(f"Cleaned up old task {task_id}")

# Global task manager instance
task_manager = TaskManager() 