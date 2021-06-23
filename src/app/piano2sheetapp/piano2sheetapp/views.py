from django.http import HttpResponse
from django.shortcuts import redirect, render
from pytube import YouTube

# def index(request):
#     #return HttpResponse('index')
#     return render(request, 'index.html')

def about(request):
    return render(request, 'about.html')
    #return HttpResponse('about')

def index(request):
    # checking whether request.method is post or not
    if request.method == "POST":

        # getting link form frontend
        link = request.POST['link']
        video = YouTube(link)

        # setting video resolution
        stream = video.streams.get_lowest_resolution()

        stream.download()

        return render(request, 'index.html')
    return render(request, 'index.html')
