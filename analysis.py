import os
from datetime import datetime
from typing import Any, Dict
import google.generativeai as genai
from google.generativeai import GenerativeModel

class ContentAnalyzer:
    def __init__(self, api_key: str, existing_session=None):
        self.api_key = api_key
        genai.configure(api_key=api_key)
        
        if existing_session:
            self.model = existing_session
        else:
            generation_config = {
                "temperature": 1,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
                "response_mime_type": "text/plain",
            }
            
            self.model = genai.GenerativeModel(
                model_name="gemini-1.5-pro",
                generation_config=generation_config,
            )
            self.model = self.model.start_chat(history=[])
        
        # Updated system prompt focused on CSV and JSON generation
        self.system_prompt = """You are a high-precision content processing system. Process the provided content into both CSV and JSON formats:

        1. CSV Format Requirements:
           - Identify key data points for table format
           - Define intuitive column headers
           - Use commas as delimiters; enclose fields with commas in double quotes
           - Maintain consistent data types across rows

        2. JSON Format Requirements:
           - Structure data hierarchically
           - Include metadata and relationships
           - Ensure proper nesting of related information
           - Maintain data type consistency

        Processing Guidelines:
        - Extract all relevant data points
        - Preserve relationships and context
        - Ensure machine-readable output
        - Include timestamps and metadata where applicable"""
    
    async def analyze_content(self, directory: str) -> Dict[str, Any]:
        try:
            # Read content
            content = ""
            for filename in os.listdir(directory):
                if filename.endswith(".md"):
                    with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
                        content += f.read() + "\n\n"

            # Create processing prompt
            processing_prompt = f"Process the following content into both CSV and JSON formats. Provide both formats separately:\n\n{content[:10000]}"
            
            # Combine prompts
            full_prompt = f"{self.system_prompt}\n\n{processing_prompt}"

            # Generate response
            response = self.model.generate_content(full_prompt)

            # Save processing details for debugging
            debug_info = {
                "timestamp": str(datetime.now()),
                "input_content_length": len(content),
                "system_prompt": self.system_prompt,
                "processing_prompt": processing_prompt,
                "full_prompt": full_prompt,
                "response": response.text,
            }

            # Save debug information
            debug_file = os.path.join(directory, "processing_debug.txt")
            with open(debug_file, 'w', encoding='utf-8') as f:
                for key, value in debug_info.items():
                    f.write(f"{key}: {value}\n")

            # Save CSV output
            csv_file = os.path.join(directory, "processed_content.csv")
            with open(csv_file, 'w', encoding='utf-8') as f:
                # Extract CSV portion from response
                if "CSV" in response.text:
                    csv_content = response.text.split("CSV")[1].split("JSON")[0].strip()
                    f.write(csv_content)

            # Save JSON output
            json_file = os.path.join(directory, "processed_content.json")
            with open(json_file, 'w', encoding='utf-8') as f:
                # Extract JSON portion from response
                if "JSON" in response.text:
                    json_content = response.text.split("JSON")[1].strip()
                    f.write(json_content)

            return {
                "success": True,
                "csv_file": csv_file,
                "json_file": json_file,
                "debug_file": debug_file
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }