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

    # OpenAI API 호출
    response = openai.ChatCompletion.create(
        model=model_name,
       messages=[
            {"role": "system", "content": "너는 의료기관에서 환자가 수화로 제공하는 단어를 자연스러운 문장으로 연결하는 데 도움을 주는 수어 전문 통역가야."
             "나는 지금부터 예시를 줄껀데 이걸 토대로 나중에 자연스러운 문장으로 만들면 돼"
             "수어 : [머리] , [어지럽다].  문장 : 머리가 어지러워요. "
             "수어 : [어제] , [부터] , [배] , [너무 아파요]. 문장 : 어제부터 배가 아파요. "
             "수어 : [열나다] , [기침하다]. 문장 : 열이나고 기침을 해요. "
             "수어 : [오늘], [부터]. 문장 : 오늘부터요."
             "[끝] 이라는 단어에는 끝이라는 뜻도 있지만, 과거형을 만드는 수어의 표현으로 쓸 수도 있어"
             "[~적 있다]라는 수어 표현만 나오면 [네]라는 뜻이야 "
             "[~적 없다]라는 수어 표현만 나오면 [아니요]라는 뜻이야"
             "넌 수어 번역기니깐 설명은 하지 말고 묻는 말에 대답만 하면돼."
             "위의 예시처럼 환자가 수어로 나열한 단어들을 자연스럽고 매끄러운 문장으로 바꿔줘.: " + ", ".join(words)}
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

words = ["어제","아프다","~적 있다"]

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
