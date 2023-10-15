# importing required modules
import os
import tempfile
from django.http import FileResponse
from PyPDF2 import PdfReader
from gtts import gTTS
from io import BytesIO

# def playAudio(audioBytes):
#     audioBytes.seek(0)  # move the BytesIO object pointer to the start
#     audio = AudioSegment.from_file(audioBytes, format='mp3')
#     play(audio)


total_content = ""


def getText(file_path, total_content):
    reader = PdfReader(file_path)
    # print(reader.numPages)
    val = len(reader.pages)
    for i in range(val):
        page = reader._get_page(i)
        page_content = page.extract_text()
        total_content += page_content

    return val, total_content



def sumToAudio(summary):
    audio = gTTS(summary)
    af = BytesIO()
    audio.write_to_fp(af)
    return os.path.abspath(af)
    # return af
