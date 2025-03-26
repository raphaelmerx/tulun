# Generated by Django 5.1.5 on 2025-02-02 05:20

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('translations', '0006_systemconfiguration'),
    ]

    operations = [
        migrations.AddField(
            model_name='systemconfiguration',
            name='post_editing_model',
            field=models.CharField(choices=[('anthropic:claude-3-5-sonnet-20241022', 'Anthropic: Claude 3.5 Sonnet'), ('google:gemini-2.0-flash-exp', 'Google: Gemini 2.0 Flash')], default='anthropic:claude-3-5-sonnet-20241022', help_text='The model used for post-editing translations', max_length=100),
        ),
    ]
