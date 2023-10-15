import pytube
import whisper
import os
from sumy.summarizers.luhn import LuhnSummarizer
# from sumy.nlp.stemmers import Stemmer
# from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from gptKaam import *

gpt_3 = GPT_3(os.getenv('OPENAI_API_KEY'))


model = whisper.load_model("base")


def urlToText(url):
    data = pytube.YouTube(url)
    audioF = data.streams.get_audio_only()
    name = audioF.download()

    model = whisper.load_model("base")
    audio_file = open(name, 'rb')
    audio_bytes = audio_file.read()
    text = model.transcribe(name)
    valuable = text['text']

    return audio_bytes, valuable


url = input("yt url")

audioF, text = urlToText(url)

# print(text)
print("url to text extracted")

# print()

summary = " "

print("summary is being calculated")

parser = PlaintextParser(text, Tokenizer('english'))
summarizer_1 = LuhnSummarizer()
summary_1 = summarizer_1(parser.document, 10)
for sentence in summary_1:
    summary += str(sentence)


lang = input("translate the video into which language?")
translated = gpt_3.translate(text, lang)


print(translated)

<<<<<<< HEAD
=======
print(summary)


question = input("question dedo : ")
context = summary 
qna_answer = gpt_3.qna(question , summary)


# print(qna_answer)
>>>>>>> 5cfd3982ffd865c01b5fe575f14ed08b277658a3
