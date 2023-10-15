from django import forms


class LearnForm(forms.Form):
    text = forms.CharField(label="What do you want to learn?", max_length=100, widget=forms.TextInput(attrs={'class': 'input-text'}))
    MY_CHOICES = (
        ('1', '1 question'),
        ('2', '2 questions'),
        ('3', '3 questions'),
        ('4', '4 questions'),
        ('5', '5 questions'),
    )
    no_of_question_for_quiz = forms.ChoiceField(choices=MY_CHOICES, widget=forms.Select(attrs={'class': 'input-quiz'}))
