from django.shortcuts import render
# Create your views here.

def home(request):
    if not request.user.is_authenticated:
        return render(request, 'enterpage.html')
    else:
        # do something for authenticated users
        return render(request, 'home.html')




