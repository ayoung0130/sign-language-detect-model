import os
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
model_name = "gpt-3.5-turbo"

def words_to_sentence(words):
    # 프롬프트 생성
    prompt = "환자가 수어로 나열한 단어들을 자연스러운 문장으로 바꿔줘: " + ", ".join(words)
    
    # OpenAI API 호출
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are an assistant who helps to connect words provided by patients in sign language into natural sentences in a medical institution."},
            {"role": "user", "content": prompt}
        ],
        n=1,    
        temperature=0.0,
        max_tokens=1024,
        top_p=1.0,
        frequency_penalty=0.0,  
        presence_penalty=0.0
    )
    sentence = response.choices[0]['message']['content'].strip()
    return sentence

words = ["너무 아프다" , "머리", "어제"]

sentence = words_to_sentence(words)
print(f"[{model_name}]: {sentence}")
