# Generated by Django 4.2 on 2023-04-08 22:58

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0001_initial'),
    ]

    operations = [
        migrations.RenameField(
            model_name='userinfo',
            old_name='first_name',
            new_name='username',
        ),
        migrations.RemoveField(
            model_name='userinfo',
            name='address',
        ),
        migrations.RemoveField(
            model_name='userinfo',
            name='last_name',
        ),
        migrations.RemoveField(
            model_name='userinfo',
            name='phone_number',
        ),
    ]
