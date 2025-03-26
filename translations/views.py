import threading
import json

from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.http import StreamingHttpResponse
from django.http import JsonResponse
from asgiref.sync import sync_to_async

from translations.utils import TranslatorGoogle, TranslatorHuggingFace
from .models import GlossaryEntry, CorpusEntry, SystemConfiguration, Translation, EvalRow

# Initialize these outside the view, using a lock for thread safety
translator = None
lock = threading.Lock()

def initialize_resources():
    global translator
    config = SystemConfiguration.load()
    with lock:
        if translator is None:
            if config.translation_model.lower() == 'google translate':
                translator = TranslatorGoogle()
            else:
                translator = TranslatorHuggingFace(config.translation_model)


def translate_view(request):
    config = SystemConfiguration.load()
    if config.login_required and not request.user.is_authenticated:
        from django.shortcuts import redirect
        return redirect('login')
    if request.headers.get('accept') == 'text/event-stream':
        # This is the SSE request
        source_text = request.GET.get('source_text')
        if translator is None:
            initialize_resources()
        
        async def event_stream():
            # Send MT translation
            mt_translation = translator.translate(source_text)
            yield f"data: {json.dumps({'type': 'mt_translation', 'data': mt_translation})}\n\n"

            # Send glossary entries
            glossary_entries = await sync_to_async(GlossaryEntry.get_entries)(source_text)
            yield f"data: {json.dumps({'type': 'glossary_entries', 'data': [gloss_entry.as_dict() for gloss_entry in glossary_entries]})}\n\n"
            
            # Send similar sentences
            similar_sentences = await sync_to_async(CorpusEntry.get_top_similar_bm25)(source_text, translator.config.num_sentences_retrieved)
            serialized_similar_sentences = [line.as_dict() for line in similar_sentences]
            yield f"data: {json.dumps({'type': 'similar_sentences', 'data': serialized_similar_sentences})}\n\n"
            
            # Send final translation
            ai_response = await translator.get_post_edited_translation(source_text, similar_sentences, glossary_entries)
            final_translation = ai_response['final_translation']
            yield f"data: {json.dumps({'type': 'final_translation', 'data': final_translation})}\n\n"

            yield f"data: {json.dumps({'type': 'corrections', 'data': [c.as_dict() for c in ai_response['corrections']]})}\n\n"

            translation = await sync_to_async(Translation.objects.create)(
                source_text=source_text,
                mt_translation=mt_translation,
                final_translation=final_translation,
                created_by=request.user if request.user.is_authenticated else None
            )
            await sync_to_async(translation.glossary_entries.set)(glossary_entries)
            await sync_to_async(translation.corpus_entries.set)(similar_sentences)

        response = StreamingHttpResponse(event_stream(), content_type='text/event-stream')
        response['Cache-Control'] = 'no-cache'
        response["X-Accel-Buffering"] = "no"
        return response

    context = {'eval_rows': [row.as_dict() for row in EvalRow.objects.all()]}
    return render(request, 'translations/translate.html', context)


@login_required
def create_corpus_entry(request):
    if not request.user.has_perm('translations.add_corpusentry'):
        return JsonResponse({'error': 'Permission denied'}, status=403)
    if request.method == 'POST':
        source_text = request.POST.get('src')
        translation_text = request.POST.get('translation')
        corpus_entry = CorpusEntry.objects.update_or_create(english_text=source_text, translated_text=translation_text)[0]
        return JsonResponse({'id': corpus_entry.id, 'source_text': corpus_entry.english_text, 'translation_text': corpus_entry.translated_text})
    return JsonResponse({'error': 'Method not allowed'}, status=405)
