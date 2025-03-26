import csv
import os
from dotenv import load_dotenv 
load_dotenv()
from typing import Optional
import google.generativeai as genai


genai.configure(api_key=os.environ["GEMINI_API_KEY"])

class GenerativeModel:
    """ A wrapper around Google AI Studio API that keeps a memory of inferences. """
    def __init__(self, client: Optional[genai.GenerativeModel] = None, **kwargs):
        if not client:
            self.client = genai.GenerativeModel('gemini-2.0-flash', **kwargs)
        else:
            self.client = client
        self.results = []
        self.filename = 'gemini_inference.csv'
        self.load_results()
    
    def load_results(self):
        """ Load previous results from a CSV file """
        if not os.path.exists(self.filename):
            print(f'No previous results found.')
            return
        with open(self.filename) as f:
            reader = csv.DictReader(f)
            self.results = [row for row in reader]
        print(f'Loaded {len(self.results)} input-output pairs.')
    
    def generate_content(self, prompt, **kwargs) -> str:
        if self.get_result_for_prompt(prompt):
            return self.get_result_for_prompt(prompt)['output']
        result: dict = self.client.generate_content(prompt, **kwargs)
        self.results.append({'input': prompt, 'output': result.text})
        self.save_results()
        return result.text
    
    def get_result_for_prompt(self, text) -> Optional[dict]:
        """ Get previous result for a given text, if any. """
        return next((r for r in self.results if r['input'] == text), None)
    
    def save_results(self):
        """ Save the results to a CSV file """
        with open(self.filename, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
            writer.writeheader()
            writer.writerows(self.results)