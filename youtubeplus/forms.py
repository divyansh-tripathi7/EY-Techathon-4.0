from django import forms


class YtForm(forms.Form):
    url = forms.CharField(label="Enter youtube URL here", max_length=100)
    lang = forms.CharField(label="Add language", max_length=20)
    MY_CHOICES = (
        ('5',  'lines 5'),
        ('10', 'lines 10'),
        ('15', 'lines 15'),
    )
    lines = forms.ChoiceField(choices=MY_CHOICES)
