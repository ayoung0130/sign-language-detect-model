import os
import openai
from dotenv import load_dotenv

#pip install speechrecognition
#pip install gTTs
#pip install playsound==1.2.2
#tts
import speech_recognition as sr
from gtts import gTTS
import os
import time
import playsound

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
model_name = "gpt-4o"

def words_to_sentence(words):
    # 프롬프트 생성
    prompt = "위의 예시처럼 환자가 수어로 나열한 단어들을 자연스럽고 매끄러운 문장으로 바꿔줘: " + ", ".join(words)
    
    # OpenAI API 호출
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are an assistant who helps to connect words provided by patients in sign language into natural sentences in a medical institution."},
            {"role": "system", "content": "i will give you some examples in korean, and you respond korean too"},
            {"role": "system", "content": "질문 : [머리] , [어지럽다].  답변 : 머리가 어지러워요. "},
            {"role": "system", "content": "질문 : [어제] , [부터] , [배] , [너무 아파요]. 답변 : 어제부터 배가 아파요. " },
            {"role": "system", "content": "질문 : [열나다] , [기침하다]. 답변 : 열이나고 기침을 해요. "},
            {"role": "system", "content": "질문 : [오늘], [부터]. 답변: 오늘부터요."},
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

words = ["월요일" , "부터" , "머리" , "목" , "너무 아파요"]

sentence = words_to_sentence(words)
print(f"[{model_name}]: {sentence}")

#tts

def speak(text):

     tts = gTTS(text=text, lang='ko')  # tts에 텍스트를 한국어로 

     filename='voice.mp3' # 파일을 만들고 

     tts.save(filename) #파일을 tts로 

     playsound.playsound(filename) # 그 파일을 실행

     os.remove(filename)  # 실행 후 제거

speak(sentence)