import os
from gtts import gTTS
from django.shortcuts import render
from .bookBuddy import getText
from .forms import BookbuddyForm
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from django.conf import settings


def bookbuddy(request):
    if request.method == "POST":
        input = BookbuddyForm(request.POST, request.FILES)
        if input.is_valid():
            print(len(request.FILES))
            file = request.FILES['file']
            val, text = getText(file, "")
            print("The summary generation has started...")
            parser = PlaintextParser(text, Tokenizer('english'))
            summarizer_1 = LuhnSummarizer()
            summary_1 = summarizer_1(parser.document, 10)
            output = ""
            for sentence in summary_1:
                output += str(sentence)
            file = text_to_audio(output)
            ctx = {
                'file': file,
                'output': output,
                'form': input,
            }
            return render(request, 'bookbuddy_block.html', ctx)
    else:
        input = BookbuddyForm()
        return render(request, 'bookbuddy.html', {'form': input})


def text_to_audio(text):
    # Create a gTTS object
    tts = gTTS(text=text)

    # Get the path of the media directory
    media_dir = os.path.join(settings.MEDIA_ROOT, 'audio')

    # Create the media directory if it doesn't exist
    os.makedirs(media_dir, exist_ok=True)

    # Create the path for the audio file
    audio_file_path = os.path.join(media_dir, 'audio.mp3')

    # Write the audio data to the file
    tts.save(audio_file_path)

    # Return the path of the audio file
    return audio_file_path
