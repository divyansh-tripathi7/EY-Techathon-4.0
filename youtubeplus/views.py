import yt_dlp
from django.conf import settings
from django.shortcuts import render
import ffmpeg
from sumy.summarizers.luhn import LuhnSummarizer
import whisper
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
import os
from gpt import GPT_3
from .forms import YtForm


gpt_3 = GPT_3(os.environ['API_KEY'])
# Create your views here.


model = whisper.load_model("base")


def download_mp3(url):
    options = {
        'format': 'bestaudio/best',
        'extractaudio': True,
        'audioformat': 'mp3',
        'outtmpl': os.path.join(settings.MEDIA_ROOT, '%(title)s.%(ext)s'),
        'getfilename': True,
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        file_path = os.path.join(settings.MEDIA_ROOT, filename)
    return filename, file_path


def urlToText(url):
    file, file_path = download_mp3(url)
    print(file_path)
    model = whisper.load_model("base")
    audio_file = open(file, 'rb')
    audio_bytes = audio_file.read()
    text = model.transcribe(file)
    valuable = text['text']
    return audio_bytes, valuable


def youtubeplus(request):
    form = YtForm(request.POST)
    if request.method == "POST":
        form = YtForm(request.POST)
        if form.is_valid():
            url = request.POST['url']
            lang = request.POST['lang']
            lines = request.POST['lines']

            print(url)
            audio, text = urlToText(url)
            summary = ""
            print(" Summary is being calculated")
            parser = PlaintextParser(text, Tokenizer('english'))
            summarizer_1 = LuhnSummarizer()
            summary_1 = summarizer_1(parser.document, lines)
            for sentence in summary_1:
                summary += str(sentence)
            translated = gpt_3.translate(summary, lang)
            ctx = {
                'translated': translated,
                'summary': summary,
                'form': form,
            }
            print(summary)
            print(translated)
            return render(request, 'youtubeplus_block.html', ctx)
    else:
        form = YtForm()
        return render(request, 'youtubeplus.html', {'form': form})
    return render(request, 'youtubeplus.html', {'form': form})


def converttovideo(request):
    stream = ffmpeg.input('input.mp4')
    stream = ffmpeg.hflip(stream)
    stream = ffmpeg.output(stream, 'output.mp4')
    ffmpeg.run(stream)
