from django import forms


class BookbuddyForm(forms.Form):
    # text = forms.CharField(widget=forms.Textarea, required=False)
    file = forms.FileField(label="Upload the file in PDF format",required=True)
    
