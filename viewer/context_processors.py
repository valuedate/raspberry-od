from django.conf import settings


def globalsettings(request):
    # return any necessary values
    return {
        'MEDIA_URL': settings.MEDIA_URL,
        'MEDIA_ROOT': settings.MEDIA_ROOT,
    }