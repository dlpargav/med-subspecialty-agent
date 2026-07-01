# Import configuration
import config

class LLMGenerator:
    """
    Orchestrates LLM query execution. Toggles between native Google Gemini SDK
    and an OpenAI-compatible API client (DeepSeek, OpenRouter, Ollama)
    based on environmental configuration.
    """
    def __init__(self):
        self.provider = config.LLM_PROVIDER
        
        # System instructions to enforce evidence-based answers and prevent hallucinations
        self.system_instruction = (
            "You are a highly precise clinical decision-support AI assisting a physician. "
            "Your task is to answer the physician's query based strictly on the provided literature snippets. "
            "\n\n"
            "CRITICAL RULES FOR RESPONDING:\n"
            "1. Grounding: Rely ONLY on the clear facts in the context snippets. If the context does not contain "
            "the answer to the question, state: 'I cannot find this information in the provided literature.' "
            "Do not attempt to answer using external training knowledge or write speculative guesses.\n"
            "2. Citations: Every statement or claim you make must cite its source. Append the snippet index "
            "number in brackets at the end of the sentence or clause, e.g., 'Kawasaki disease is treated with IVIG [Snippet #1].' "
            "Never make an uncited statement if it is based on the text.\n"
            "3. Clinical Tone: Maintain a highly professional, objective, and clear clinical tone. Avoid conversational filler."
        )

        if self.provider == "gemini":
            from google import genai
            if not config.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY is missing in your configuration.")
            self.client = genai.Client(api_key=config.GEMINI_API_KEY)
            self.model_name = "gemini-1.5-flash"
            
        elif self.provider == "openai_compatible":
            from openai import OpenAI
            # Set up client pointing to local Ollama or cloud providers (DeepSeek, OpenRouter)
            self.client = OpenAI(
                base_url=config.OPENAI_API_BASE,
                api_key=config.OPENAI_API_KEY or "no-key"
            )
            self.model_name = config.OPENAI_MODEL_NAME
        else:
            raise ValueError(f"Unknown LLM_PROVIDER: {self.provider}")

    def generate_response(self, question: str, context: str) -> str:
        """
        Submits the question and matching context to the LLM and retrieves the response.
        """
        # User prompt structure
        user_prompt = (
            f"Physician Question:\n{question}\n\n"
            f"Literature Context Snippets:\n{context}\n\n"
            f"Please write your clinical answer below, adhering strictly to the citation rules."
        )
        
        try:
            if self.provider == "gemini":
                from google.genai import types
                
                # Make API call using the official Google Gen AI SDK
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=self.system_instruction,
                        temperature=0.0,  # Zero temperature for deterministic grounding
                        max_output_tokens=1024
                    )
                )
                return response.text
                
            elif self.provider == "openai_compatible":
                # Make API call using OpenAI client
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.system_instruction},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.0,
                    max_tokens=1024
                )
                return response.choices[0].message.content
                
        except Exception as e:
            return f"Error communicating with LLM Provider ({self.provider}): {str(e)}"
