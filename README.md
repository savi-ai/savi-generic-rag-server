# Savi Generic RAG Server

A comprehensive RAG (Retrieval-Augmented Generation) server with support for multiple LLM providers, document processing, and advanced agentic capabilities.

## Features

- **Document Processing**: Support for PDF, DOCX, TXT, and XLSX files
- **Vector Database**: ChromaDB integration with configurable chunking
- **Multiple LLM Providers**: OpenAI, Anthropic, and Ollama support
- **Agentic RAG**: Advanced reasoning capabilities with ReAct and Self-Critic agents
- **Evaluation Framework**: Comprehensive evaluation of RAG responses
- **Guardrails**: Content filtering and safety measures
- **RESTful API**: FastAPI-based backend with comprehensive endpoints

## Self-Critic Agent

The Self-Critic Agent is an advanced agentic RAG component that implements a self-improvement workflow using LangGraph. It works by:

1. **Generating Initial Response**: Creates a first response to the user's query
2. **Self-Evaluation**: Critically analyzes the response for accuracy, completeness, and quality
3. **Iterative Improvement**: Generates improved responses based on self-criticism
4. **Final Output**: Delivers the best possible response after multiple improvement cycles

### Key Benefits

- **Quality Assurance**: Automatically identifies and fixes response issues
- **Self-Learning**: Continuously improves responses through self-reflection
- **Configurable**: Adjustable iteration limits and improvement criteria
- **LangGraph Integration**: Built on robust workflow orchestration framework

### Usage

```python
from app.services.self_critic_agent import SelfCriticAgent
from app.services.llm_service import LLMService

# Initialize services
llm_service = LLMService()

# Create self-critic agent
agent = SelfCriticAgent(
    llm_service=llm_service,
    query="Your question here",
    context=[{"content": "Relevant context", "metadata": {"filename": "doc.pdf"}}],
    system_prompt="You are a helpful AI assistant.",
    agentic_config={"max_iterations": 3}
)

# Execute the workflow
result = await agent.execute()
print(f"Final response: {result['response']}")
print(f"Was improved: {result['was_improved']}")
print(f"Iterations: {result['iteration_count']}")
```

### Configuration

The self-critic agent supports various configuration options:

- `max_iterations`: Maximum number of improvement cycles (default: 3)
- `llm_parameters`: Custom LLM settings (temperature, model, etc.)
- `system_prompt`: Custom system instructions for the agent

## Installation

1. Clone the repository:
```bash
git clone https://github.com/savi-ai/savi-generic-rag-server.git
cd savi-generic-rag-server
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp env.example .env
# Edit .env with your configuration
```

5. Start the server:
```bash
python run.py
```

## API Endpoints

### Core RAG Operations

- `POST /savi-rag-api/api/upload` - Upload and process documents
- `POST /savi-rag-api/api/test` - Test RAG queries with agentic capabilities
- `GET /savi-rag-api/api/usecases` - List available use cases

### Agentic RAG

- **Self-Critic Mode**: Enable with `agentic_config={"type": "self_critic"}`
- **ReAct Mode**: Enable with `agentic_config={"type": "react"}`

### Evaluation

- `POST /savi-rag-api/api/evaluate` - Start evaluation tasks
- `GET /savi-rag-api/api/evaluation/status/{task_id}` - Check evaluation status
- `GET /savi-rag-api/api/evaluation/download/{task_id}` - Download results

## Configuration

### LLM Providers

Configure your preferred LLM provider in the `.env` file:

```env
# OpenAI
OPENAI_API_KEY=your_key_here
DEFAULT_LLM_PROVIDER=openai

# Anthropic
ANTHROPIC_API_KEY=your_key_here
DEFAULT_LLM_PROVIDER=anthropic

# Ollama (local)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
DEFAULT_LLM_PROVIDER=ollama
```

### Vector Database

- **ChromaDB**: Default vector store with configurable persistence
- **Chunking**: Configurable chunk size and overlap
- **Embeddings**: Sentence transformers with customizable models

## Testing

Run the comprehensive test suite:

```bash
# Test self-critic agent
python test_self_critic_agent.py

# Test agentic service integration
python test_agentic_integration.py

# Run comprehensive tests
python test_comprehensive_self_critic.py
```

## Architecture

The system is built with a modular architecture:

- **Services Layer**: Core business logic (LLM, Vector, Document, Agentic)
- **API Layer**: FastAPI endpoints with comprehensive error handling
- **Workflow Engine**: LangGraph-based agent orchestration
- **Storage Layer**: ChromaDB for vectors, SQLite for metadata

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request


## Support

For questions and support, please [create an issue](link-to-issues) 
