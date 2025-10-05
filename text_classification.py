import os
import dotenv
import argparse
from pydoc import text 
import openai
import pandas as pd
import time

dotenv.load_dotenv('google.env', override=True) # Loads environment variables from .env file
client = openai.OpenAI(api_key=os.getenv('api_key'),
                       base_url=os.getenv('base_url')) # Initializes OpenAI client

def create_prompt(text): # Generates classification prompts
    ''' Generates prompt for sentiment classification.
    Args:
    text: classify this text.
    Returns:
    input for LLM.
    '''
    task = 'Is the sentiment positive or negative?'
    answer_format = 'Answer (''Positive''/''Negative'')'
    return f'{text}\n{task}\n{answer_format}:'


def call_llm(prompt): # Calls the large language model
    ''' Query large language model and return answer.
    Args:
    prompt: input prompt for language model.
    Returns:
    Answer by language model.
    '''
    for nr_retries in range(1, 4):
        try:
            response = client.chat.completions.create(
            model=os.getenv('model'),
            messages=[
            {'role':'user', 'content':prompt}
            ]
            )
            return response.choices[0].message.content
        except:
            time.sleep(nr_retries * 2)
    raise Exception('Cannot query OpenAI model!')

def classify(text): # Classifies one text document
    ''' Classify input text.
    Args:
    text: assign this text to a class label.
    Returns:
    name of class.
    '''
    prompt = create_prompt(text)
    label = call_llm(prompt)
    return label

if __name__ == '__main__': # Reads text, classifies, and writes result
    parser = argparse.ArgumentParser() #  Defines command-line arguments
    parser.add_argument('file_path', type=str, help='Path to input file')
    args = parser.parse_args()
    df = pd.read_csv(args.file_path) # Reads input
    df['class'] = df['text'].apply(classify) #  Classifies text
    statistics = df['class'].value_counts() # Generates output
    print(statistics)
    df.to_csv('result.csv')