import os
import openai
import os
import playsound
from dotenv import load_dotenv
from gtts import gTTS

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
model_name = "gpt-4o"

def words_to_sentence(words):

    # OpenAI API 호출
    response = openai.ChatCompletion.create(
        model=model_name,
       messages=[
            {"role": "system", "content": 
                    "너는 의료기관에서 환자가 수어로 제공하는 단어를 자연스러운 문장으로 바꿔주는 전문 수어 통역가야." +
                    "너에게는 문자 리스트가 주어질 거야. 이 리스트는 모델이 예측한 단어 리스트야." +
                    "예시를 보여줄게. 이것을 참고해서 단어 정답 리스트를 자연스러운 문장으로 만들면 돼." +

                    "정답 리스트에 따른 문장 예문을 보여줄게. [끝] 이라는 단어는 문장을 과거형으로 만들어." +
                    "수어: [아래] -> 문장: 아래요." +
                    "수어: [머리, 어지럽다] -> 문장: 머리가 어지러워요." +
                    "수어: [어제, 부터, 배, 아파요] -> 문장: 어제부터 배가 아파요." +
                    "수어: [열나다, 창백하다] -> 문장: 열이나고 창백해요." +
                    "수어: [월요일, 부터] -> 문장: 월요일부터요." +
                    "수어: [다리, 깔리다, 뼈, 부러지다, 끝] -> 문장: 다리가 깔려서 뼈가 부러졌어요." +
                    "수어: [월요일, 기절, 끝] -> 문장: 월요일에 기절을 했어요." +
                    "수어: [네] -> 문장: 네." +
                    "수어: [아니오] -> 문장: 아니오." +
                    "수어: [알려주세요] -> 문장: 알려주세요." +

                    "정답 리스트에 없는 다른 단어를 임의로 포함하지마" +
                    "넌 수어 통역가이므로 자세한 설명은 필요 없어. 자연스러운 문장만 알려줘" +
                    "이제, 위의 예시처럼 환자가 수어로 나열한 단어들을 자연스럽고 매끄러운 문장으로 바꿔줘: ".join(words)}
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

def tts(text):

     tts = gTTS(text=text, lang='ko')  # tts에 텍스트를 한국어로 

     filename='voice.mp3' # 파일을 만들고 

     tts.save(filename) #파일을 tts로 

     playsound.playsound(filename) # 그 파일을 실행

     os.remove(filename)  # 실행 후 제거
