#!/usr/bin/env python3
"""
Savi RAG Backend - Run Script
"""

import uvicorn
import os
from dotenv import load_dotenv

def main():
    """Main entry point for the application"""
    # Load environment variables
    load_dotenv()
    
    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("DEBUG", "false").lower() == "true"
    
    print(f"Starting Savi RAG Backend...")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Debug mode: {reload}")
    print(f"API Documentation: http://{host}:{port}/docs")
    
    # Run the application
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

if __name__ == "__main__":
    main() 