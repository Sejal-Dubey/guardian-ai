import requests
import json
import os

def call_gemini_api(prompt: str, api_key: str):
    """
    Calls the Gemini API with a given prompt and API key.

    Args:
        prompt (str): The text prompt to send to the model.
        api_key (str): Your Gemini API key.

    Returns:
        str: The generated text content from the model, or an error message.
    """
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent"
    headers = {
        'Content-Type': 'application/json',
        'X-goog-api-key': api_key
    }
    data = {
        "contents": [{"parts": [{"text": prompt}]}]
    }

    print("INFO: Sending request to Gemini API...")
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        response_json = response.json()

        # --- CORRECTED AND MORE ROBUST PARSING LOGIC ---
        # Check if the response has the expected structure before trying to access it
        if ('candidates' in response_json and
                len(response_json['candidates']) > 0 and
                'content' in response_json['candidates'][0] and
                'parts' in response_json['candidates'][0]['content'] and
                len(response_json['candidates'][0]['content']['parts']) > 0):
            
            generated_text = response_json['candidates'][0]['content']['parts'][0]['text']
            return generated_text
        else:
            # Handle cases where the response is valid but doesn't contain text
            print("ERROR: The API response did not contain the expected text content.")
            print(f"Full Response JSON: {response_json}")
            return None
        # --- END OF CORRECTION ---

    except requests.exceptions.RequestException as e:
        print(f"ERROR: An error occurred while making the API request: {e}")
        if 'response' in locals():
            print(f"Server Response: {response.text}")
        return None

# --- Example Usage ---
if __name__ == "__main__":
    gemini_api_key = os.environ.get("GEMINI_API_KEY", "AIzaSyDZ21jqGvrnsDF1ZMkm1JxzBsFQ5dedDns")

    if gemini_api_key == "YOUR_GEMINI_API_KEY":
        print("WARNING: Please replace 'YOUR_GEMINI_API_KEY' with your actual Gemini API key.")
    else:
        user_prompt = "Explain how AI works in a few words"
        result = call_gemini_api(user_prompt, gemini_api_key)

        if result:
            print("\n--- ✅ Gemini API Response ---")
            print(result)
