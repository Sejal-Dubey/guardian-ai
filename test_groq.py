#!/usr/bin/env python3
"""
Test Groq API Model Configuration
This script will verify if your Groq model is configured correctly
"""
import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

def test_groq_model():
    """Test if the configured Groq model works"""
    
    print("=" * 60)
    print("GROQ MODEL CONFIGURATION TEST")
    print("=" * 60)
    
    # Get configuration
    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    
    print(f"\n📋 Current Configuration:")
    print(f"   GROQ_API_KEY: {'✅ Set' if api_key else '❌ Not set'}")
    print(f"   GROQ_MODEL: {model}")
    
    if not api_key:
        print("\n❌ GROQ_API_KEY not found in .env file!")
        return False
    
    # Initialize Groq client
    client = Groq(api_key=api_key)
    
    # Test the model
    print(f"\n🧪 Testing model: {model}")
    print("-" * 40)
    
    test_prompt = "Say 'Hello, I'm working!' in exactly 5 words."
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": test_prompt}
            ],
            max_tokens=20,
            temperature=0.1
        )
        
        result = response.choices[0].message.content
        print(f"✅ SUCCESS! Model responded: {result}")
        return True
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ FAILED! Error: {error_msg}")
        
        # Parse error for common issues
        if "model not found" in error_msg.lower() or "404" in error_msg:
            print("\n⚠️  This model doesn't exist on Groq!")
            print("\n📝 Valid Groq models you can use:")
            print("   - llama-3.1-70b-versatile")
            print("   - llama-3.1-8b-instant")
            print("   - llama3-70b-8192")
            print("   - llama3-8b-8192")
            print("   - mixtral-8x7b-32768")
            print("   - gemma2-9b-it")
            print("   - gemma-7b-it")
            
            print("\n🔧 To fix, update your .env file:")
            print('   GROQ_MODEL="llama-3.1-70b-versatile"')
            
        elif "api key" in error_msg.lower() or "401" in error_msg:
            print("\n⚠️  API key is invalid or expired!")
            print("   Get a new key from: https://console.groq.com/keys")
            
        elif "rate limit" in error_msg.lower() or "429" in error_msg:
            print("\n⚠️  Rate limit exceeded. Wait a moment and try again.")
            
        else:
            print("\n⚠️  Unknown error. Check your configuration.")
        
        return False

def list_available_models():
    """List all available Groq models"""
    
    print("\n" + "=" * 60)
    print("AVAILABLE GROQ MODELS")
    print("=" * 60)
    
    models = {
        "llama-3.1-70b-versatile": "Most capable, best for complex tasks",
        "llama-3.1-8b-instant": "Fast responses, good for simple tasks",
        "llama3-70b-8192": "Previous gen, still very capable",
        "llama3-8b-8192": "Previous gen, fast",
        "mixtral-8x7b-32768": "Good for code and technical content",
        "gemma2-9b-it": "Google's model, good general purpose",
        "gemma-7b-it": "Smaller Google model, fast"
    }
    
    print("\nRecommended for your email security system:")
    for model, description in models.items():
        print(f"\n📌 {model}")
        print(f"   {description}")
    
    print("\n💡 Best choice for email analysis: llama-3.1-70b-versatile")

def quick_fix():
    """Provide quick fix for .env file"""
    
    print("\n" + "=" * 60)
    print("QUICK FIX")
    print("=" * 60)
    
    print("\n1. Open your .env file")
    print("2. Find the line with GROQ_MODEL")
    print("3. Replace it with:")
    print('   GROQ_MODEL="llama-3.1-70b-versatile"')
    print("4. Save the file")
    print("5. Restart your backend")
    
    print("\n📝 Complete .env example:")
    print("-" * 40)
    print('GROQ_API_KEY="your-api-key-here"')
    print('GROQ_MODEL="llama-3.1-70b-versatile"')
    print('# ... other settings ...')

if __name__ == "__main__":
    # Run the test
    success = test_groq_model()
    
    if not success:
        # Show available models if test failed
        list_available_models()
        quick_fix()
    else:
        print("\n✅ Your Groq configuration is working perfectly!")
        print("   The 400 errors in your logs might be from:")
        print("   - Rate limiting (too many requests)")
        print("   - Token limit exceeded (messages too long)")
        print("   - Malformed JSON in prompts")
    
    print("\n" + "=" * 60)