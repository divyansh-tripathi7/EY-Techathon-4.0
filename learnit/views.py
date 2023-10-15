import os
from django.shortcuts import render
from gpt import GPT_3, load_dotenv
from .forms import LearnForm
import html
from django.http import HttpResponse
from django.template.defaultfilters import linebreaksbr
# Create your views here.


load_dotenv()
gpt_3 = GPT_3(os.environ['API_KEY'])


def learnit(request):
    if request.method == "POST":
        form = LearnForm(request.POST)
        if form.is_valid():
            text = request.POST['text']
            question = request.POST['no_of_question_for_quiz']
            result = gpt_3.teach(200, text)
            print(result)
            resource = gpt_3.resources2(text)
            print(resource)
            quiz = gpt_3.QuizMe(text, question)
            print(quiz)
            print(type(quiz))
            ctx = {
                'quiz': quiz,
                'resource': resource,
                'desc': result,
                'linebreaksbr': linebreaksbr,
                'form': form,

            }
            return render(request, 'learnit_block.html', ctx)
    else:
        form = LearnForm()
        return render(request, 'learnit.html', {'form': form})
