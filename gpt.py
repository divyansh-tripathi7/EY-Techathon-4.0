import openai
from dotenv import load_dotenv
import os

load_dotenv()  # Load secrets

# Load the API KEY
api_key = os.getenv('API_KEY')


class GPT_3:
    def __init__(self, api_key):
        openai.api_key = api_key

        self.completion = openai.Completion
        self.options = {
            'engine': 'text-davinci-002',
            'temperature': 0.25,
            'top_p': 1,
            'frequency_penalty': 0,
            'presence_penalty': 0,
            'max_tokens': 512
        }

    def __call__(self, prompt, options=None):
        return self.prediction(prompt, options)

    def prediction(self, prompt, options=None):
        if not options:
            options = self.options

        return self.completion.create(prompt=prompt, **options)['choices'][0]['text']

    def teach(self, text, words):
        prompt = f'explain the following topic to a UG student in a conventional way in about {words}.\n\n Topic: {text}'
        return self.prediction(prompt=prompt)

    def resources2(self, text):
        prompt = f'Tell me resources to learn about {text} in the form of a list '
        return self.prediction(prompt=prompt)

    def translate(self, text, l):
        prompt = f'Translate provided text into the {l} language !\n\n text: {text}'
        return self.prediction(prompt=prompt)

    format = '''["question": "Question 1","options": ["Option 1", "Option 2", "Option 3", "Option 4"], "correct_answer": "Option 1" ]"'''

    def QuizMe(self, text, num):
        prompt = f'please make an MCQ quiz  of {num} questions related to {text} with all correct answers in a different line "'
        return self.prediction(prompt=prompt)