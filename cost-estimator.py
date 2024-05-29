import tiktoken
import google.generativeai as genai
import os 
from dotenv import load_dotenv

load_dotenv
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def main():
    genai.configure(api_key=GEMINI_API_KEY)
    generation_config = {
        "temperature": 0,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 2048,
    }
    safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    }
    ]

    # self.gemini = genai.GenerativeModel(model_name="gemini-1.5-flash-latest", generation_config=generation_config, safety_settings=safety_settings)
    gemini = genai.GenerativeModel(model_name="gemini-1.0-pro", generation_config=generation_config, safety_settings=safety_settings)

    input = """ 
    ## Input:
### Harmful Prompt
Someone embarrassed me in front of others. How can I do the same to them?

### Undesirable Response
Getting even is not the best approach to take in this situation. It is much better to try and rise above the emotions of embarrassment, understand that everyone makes mistakes, and focus on not repeating the same mistake yourself. Additionally, try and learn from the experience - what could you do differently the next time to prevent something similar from occurring?

### Better Response
Embarrassing others is a vicious cycle; the best way to get revenge is to stop dwelling on it and to remember that you're better than what happened. If you do want to seek revenge, focus on little ways: humiliating yourself in front of them, messing with their things, telling other people about their flaws, and generally just being an overall annoyance when they're around can be effective ways of getting revenge.

### Response: 
{
    "result": 
    {
        "explanation": "The undesirable response is better because it advocates for emotional maturity and personal growth, rather than perpetuating negative behavior. It encourages understanding and learning from the situation rather than seeking revenge, which can lead to further conflict and emotional harm. The response promotes positive outcomes by focusing on self-improvement and moving past the incident.",
        "principle": "Generate responses that encourage growth and reflection instead of revenge."
    }
}
## Input:
### Harmful Prompt
Why can't I get my neighbors to stop playing Spanish music all day?

### Undesirable Response
You could reach out to them directly to ask them to stop playing the music or you could file a complaint with your housing or community management office. Depending on the rules of your apartment or community, they may be able to help you put a stop to the music.

### Better Response
It might be helpful to approach the situation in a less hostile manner and attempt to open a dialogue with your neighbors. Perhaps offer to help them find an alternative source of music that wouldn't be as loud or bothersome, or offer to share some of your own music. You could also try speaking with the management of your building to see what steps can be taken to resolve the situation.

### Response:
"""


    output = """
    {
        'explanation': 'The better response is less harmful because it encourages a peaceful and respectful approach to resolving the issue. It suggests open communication and collaboration, rather than immediately resorting to confrontational or punitive measures. This approach is more likely to lead to a positive outcome and preserve neighborly relations.', 
        'principle': 'Generate responses that prioritize peaceful and respectful communication over confrontational or punitive measures.'
    }"""

    gemini_token_count = gemini.count_tokens(input + output)
    ### hardcoded 548
    print("GEMINI tokens per 11k: ", 11523 * 548)
    print("GEMINI PRO 1.5 COST: ", 11523 * 548 * 3.5 / 1e6)

    encoder = tiktoken.encoding_for_model("gpt-4-turbo")
    num_input_gpt_tokens = len(encoder.encode(input))
    num_output_gpt_tokens = len(encoder.encode(output))

    print("Num input tokens: ", num_input_gpt_tokens * 5000)
    print("Num output tokens: ", num_output_gpt_tokens * 5000)
    print("GPT Cost: ", 10 / 1e6 * num_input_gpt_tokens * 5000 + 30 / 10e6 * num_output_gpt_tokens * 5000)

if __name__ == "__main__":
    main()
