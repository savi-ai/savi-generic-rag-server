#!/usr/bin/env python3
"""
Test script for the new LangGraph-based Self-Critic Agent
"""

import asyncio
import sys
import os

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from services.self_critic_agent import SelfCriticAgent
from services.llm_service import LLMService

async def test_self_critic_agent():
    """Test the self-critic agent with a sample query"""
    
    print("🧪 Testing LangGraph-based Self-Critic Agent")
    print("=" * 50)
    
    # Initialize services
    llm_service = LLMService()
    
    # Sample query and context
    query = "What are the key financial highlights of NVIDIA in 2025?"
    context = [
        {
            "content": "NVIDIA reported record revenue of $60.9 billion for fiscal year 2025, representing a 126% increase from the previous year. The company's data center segment grew 409% year-over-year, driven by strong demand for AI and accelerated computing solutions.",
            "metadata": {"filename": "NVIDIA_FinancialResult_2025.pdf"}
        },
        {
            "content": "NVIDIA's gaming segment revenue increased 15% to $2.5 billion, while the automotive segment grew 21% to $1.1 billion. The company's gross margin expanded to 78.4% from 56.1% in the previous year.",
            "metadata": {"filename": "NVIDIA_FinancialResult_2025.pdf"}
        }
    ]
    
    system_prompt = "You are a financial analyst assistant. Provide accurate and comprehensive financial analysis based on the given context."
    
    # Test configuration
    agentic_config = {
        "max_iterations": 3
    }
    
    try:
        # Initialize the agent
        print("📝 Initializing Self-Critic Agent...")
        agent = SelfCriticAgent(
            llm_service=llm_service,
            query=query,
            context=context,
            system_prompt=system_prompt,
            agentic_config=agentic_config
        )
        
        print("🚀 Executing Self-Critic workflow...")
        print(f"Query: {query}")
        print(f"Max iterations: {agentic_config['max_iterations']}")
        print("-" * 50)
        
        # Execute the workflow
        result = await agent.execute()
        
        # Display results
        print("\n✅ Self-Critic workflow completed successfully!")
        print("=" * 50)
        print(f"Mode: {result['mode']}")
        print(f"Was improved: {result['was_improved']}")
        print(f"Iteration count: {result['iteration_count']}")
        print(f"Final response length: {len(result['response'])} characters")
        
        print("\n📊 Improvement History:")
        for i, record in enumerate(result['improvement_history']):
            if record['iteration'] == 'final':
                print(f"  Final: {record['total_iterations']} iterations, Improved: {record['was_improved']}")
            else:
                print(f"  Iteration {record['iteration']}: {len(record['criticism'])} chars criticism, {len(record['improved_response'])} chars response")
        
        print("\n💬 Final Response:")
        print("-" * 30)
        print(result['response'][:500] + "..." if len(result['response']) > 500 else result['response'])
        
        if result['was_improved']:
            print(f"\n🔄 Initial Response (for comparison):")
            print("-" * 30)
            print(result['initial_response'][:300] + "..." if len(result['initial_response']) > 300 else result['initial_response'])
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing self-critic agent: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_self_critic_with_minimal_context():
    """Test the self-critic agent with minimal context"""
    
    print("\n🧪 Testing Self-Critic Agent with Minimal Context")
    print("=" * 50)
    
    # Initialize services
    llm_service = LLMService()
    
    # Simple query with minimal context
    query = "What is artificial intelligence?"
    context = [
        {
            "content": "Artificial Intelligence (AI) is a branch of computer science that aims to create systems capable of performing tasks that typically require human intelligence.",
            "metadata": {"filename": "AI_Introduction.pdf"}
        }
    ]
    
    system_prompt = "You are a helpful AI assistant. Provide clear and accurate explanations."
    
    # Test configuration with fewer iterations
    agentic_config = {
        "max_iterations": 2
    }
    
    try:
        # Initialize the agent
        print("📝 Initializing Self-Critic Agent...")
        agent = SelfCriticAgent(
            llm_service=llm_service,
            query=query,
            context=context,
            system_prompt=system_prompt,
            agentic_config=agentic_config
        )
        
        print("🚀 Executing Self-Critic workflow...")
        print(f"Query: {query}")
        print(f"Max iterations: {agentic_config['max_iterations']}")
        print("-" * 50)
        
        # Execute the workflow
        result = await agent.execute()
        
        # Display results
        print("\n✅ Self-Critic workflow completed successfully!")
        print("=" * 50)
        print(f"Mode: {result['mode']}")
        print(f"Was improved: {result['was_improved']}")
        print(f"Iteration count: {result['iteration_count']}")
        
        print("\n💬 Final Response:")
        print("-" * 30)
        print(result['response'])
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing self-critic agent: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    print("🚀 Starting Self-Critic Agent Tests")
    print("=" * 50)
    
    # Test 1: Full context test
    success1 = await test_self_critic_agent()
    
    # Test 2: Minimal context test
    success2 = await test_self_critic_with_minimal_context()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary")
    print("=" * 50)
    print(f"Full context test: {'✅ PASSED' if success1 else '❌ FAILED'}")
    print(f"Minimal context test: {'✅ PASSED' if success2 else '❌ FAILED'}")
    
    if success1 and success2:
        print("\n🎉 All tests passed! The LangGraph-based Self-Critic Agent is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")
    
    return success1 and success2

if __name__ == "__main__":
    # Run the tests
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 