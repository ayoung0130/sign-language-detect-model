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
            {"role": "system", "content": "너는 의료기관에서 환자가 수어로 제공하는 단어를 자연스러운 문장으로 연결하는 데 도움을 주는 전문 수어 통역가야."
             "지금부터 너에게 예시를 줄 건데 이것을 토대로 나중에 자연스러운 문장으로 만들면 돼."
             "수어 : [머리], [어지럽다].  문장 : 머리가 어지러워요."
             "수어 : [어제], [부터], [배], [너무 아파요]. 문장 : 어제부터 배가 아파요."
             "수어 : [열나다], [기침하다]. 문장 : 열이나고 기침을 해요."
             "수어 : [오늘], [부터]. 문장 : 오늘부터요."
             "[끝] 이라는 단어에는 '끝'이라는 뜻도 있지만, 문장을 과거형으로 만들 수 있어. 예시를 들어줄게."
             "수어 : [월요일], [기절], [끝]. 문장 : 월요일에는 기절을 했었어요."
             "[~적 있다]라는 수어 표현만 나오면 [네]라는 뜻이야."
             "[~적 없다]라는 수어 표현만 나오면 [아니요]라는 뜻이야."
             "넌 수어 통역가이므로 자세한 설명은 하지 말고 자연스러운 문장만 출력하면 돼."
             "이제, 위의 예시처럼 환자가 수어로 나열한 단어들을 자연스럽고 매끄러운 문장으로 바꿔줘.: " + ", ".join(words)}
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

     
words = ["어제", "어지러움", "아프다", "끝"]
sentence = words_to_sentence(words)

print(f"입력된 단어: {words}")
print(f"문장 변환 결과 [{model_name}]: {sentence}")

tts(sentence)