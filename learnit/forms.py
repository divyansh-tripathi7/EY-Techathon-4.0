from django import forms


class LearnForm(forms.Form):
    text = forms.CharField(label="What do you want to learn?", max_length=100, widget=forms.TextInput(attrs={'class': 'input-text'}))
    MY_CHOICES = (
        ('5',  '5 questions'),
        ('10', '10 questions'),
        ('15', '15 questions'),
        ('20', '20 questions'),
        ('25', '25 questions'),
    )
    no_of_question_for_quiz = forms.ChoiceField(choices=MY_CHOICES, widget=forms.Select(attrs={'class': 'input-quiz'}))
