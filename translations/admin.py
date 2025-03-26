import csv
from django.contrib import admin, messages
from django import forms
from django.urls import reverse, path
from django.shortcuts import redirect, render
from django.contrib.auth.forms import (
    AdminPasswordChangeForm,
    AdminUserCreationForm,
    UserChangeForm,
)
from django.contrib.admin.views.decorators import staff_member_required

from .models import GlossaryEntry, CorpusEntry, SystemConfiguration, Translation, CustomUser, EvalRow

@admin.register(GlossaryEntry)
class GlossaryEntryAdmin(admin.ModelAdmin):
    list_display = ('english_key', 'translated_entry', 'created_at')
    search_fields = ('english_key', 'translated_entry')
    list_filter = ('created_at',)
    date_hierarchy = 'created_at'
    ordering = ('-created_at',)
    change_list_template = 'translations/glossary_changelist.html'

    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path('import-glossary/', self.import_csv, name='glossary_import_csv'),
        ]
        return custom_urls + urls

    def import_csv(self, request):
        if not request.user.is_staff:
            return redirect("..")
        if request.method == "POST":
            csv_file = request.FILES["csv_file"]
            reader = csv.DictReader(csv_file.read().decode('utf-8').splitlines())
            rows = list(reader)
            # dedup using the 'en' column
            rows = {row['en']: row for row in rows}.values()
            delete_existing = request.POST.get("delete_existing", False)
            if delete_existing:
                GlossaryEntry.objects.all().delete()
            GlossaryEntry.objects.bulk_create([
                GlossaryEntry(
                    english_key=row['en'],
                    translated_entry=row['tgt']
                ) for row in rows
            ])
            self.message_user(request, "Your csv file has been imported")
            return redirect("..")
        form = ImportCorpusEntryForm()
        payload = {"form": form}
        return render(
            request, "translations/corpusentry_import_csv.html", payload
        )

class ImportCorpusEntryForm(forms.Form):
    csv_file = forms.FileField(help_text='The CSV file should have columns "en" and "tgt"')
    delete_existing = forms.BooleanField(
        required=False,
        help_text='Check this box to delete all existing entries before importing'
    )


@admin.register(CorpusEntry)
class CorpusEntryAdmin(admin.ModelAdmin):
    list_display = ('english_text', 'translated_text', 'created_at')
    search_fields = ('english_text', 'translated_text')
    list_filter = ('created_at', 'source')
    date_hierarchy = 'created_at'
    ordering = ('-created_at',)

    change_list_template = 'translations/corpusentry_changelist.html'

    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path('import-csv/', self.import_csv, name='corpus_entry_import_csv'),
        ]
        return custom_urls + urls

    def import_csv(self, request):
        if not request.user.is_staff:
            return redirect("..")
        if request.method == "POST":
            csv_file = request.FILES["csv_file"]
            reader = csv.DictReader(csv_file.read().decode('utf-8').splitlines())
            rows = list(reader)
            # dedup using the 'en' column
            rows = {row['en']: row for row in rows}.values()
            delete_existing = request.POST.get("delete_existing", False)
            if delete_existing:
                CorpusEntry.objects.all().delete()
            for row in rows:
                CorpusEntry.objects.create(
                    english_text=row['en'],
                    translated_text=row['tgt'],
                    source=f'csv import from file {csv_file.name}'
                )
            self.message_user(request, "Your csv file has been imported")
            return redirect("..")
        form = ImportCorpusEntryForm()
        payload = {"form": form}
        return render(
            request, "translations/corpusentry_import_csv.html", payload
        )

@admin.register(Translation)
class Translation(admin.ModelAdmin):
    list_display = ('source_text', 'final_translation', 'created_by', 'created_at', 'num_TM')
    search_fields = ('source_text', 'final_translation')
    date_hierarchy = 'created_at'
    ordering = ('-created_at',)
    readonly_fields = ['source_text', 'mt_translation', 'final_translation', 'glossary_entries', 'corpus_entries', 'created_by']

    def num_TM(self, obj):
        return obj.corpus_entries.count()


@admin.register(SystemConfiguration)
class SystemConfigurationAdmin(admin.ModelAdmin):
    readonly_fields = ['created_at', 'updated_at']
    
    def has_add_permission(self, request):
        # Prevent adding if an instance already exists
        return not SystemConfiguration.objects.exists()

    def has_delete_permission(self, request, obj=None):
        # Prevent deletion of the only configuration
        return False

    def changelist_view(self, request, extra_context=None):
        # Redirect to the edit page of the first object
        try:
            config = SystemConfiguration.objects.first()
            if config:
                return redirect(reverse('admin:translations_systemconfiguration_change', args=[config.id]))
        except SystemConfiguration.DoesNotExist:
            pass
        return super().changelist_view(request, extra_context)

@admin.register(CustomUser)
class CustomUserAdmin(admin.ModelAdmin):
    add_form_template = "admin/auth/user/add_form.html"
    change_user_password_template = None
    fieldsets = (
        (None, {"fields": ("email", "password")}),
        ("Permissions", {
            "fields": ("is_active", "is_staff", "is_superuser", "groups", "user_permissions"),
        }),
        ("Important dates", {"fields": ("last_login", "date_joined")}),
    )
    add_fieldsets = (
        (None, {
            "classes": ("wide",),
            "fields": ("email", "password1", "password2"),
        }),
    )
    form = UserChangeForm
    add_form = AdminUserCreationForm
    change_password_form = AdminPasswordChangeForm
    list_display = ("email", "is_staff", "is_active", "last_login")
    list_filter = ("is_staff", "is_superuser", "is_active", "groups") 
    search_fields = ("email",)
    ordering = ("-date_joined",)
    filter_horizontal = ("groups", "user_permissions",)
    readonly_fields = ("date_joined", "last_login")

    def get_fieldsets(self, request, obj=None):
        if not obj:
            return self.add_fieldsets
        return super().get_fieldsets(request, obj)

    def get_form(self, request, obj=None, **kwargs):
        defaults = {}
        if obj is None:
            defaults["form"] = self.add_form
        defaults.update(kwargs)
        return super().get_form(request, obj, **defaults)

@admin.register(EvalRow)
class EvalRowAdmin(admin.ModelAdmin):
    list_display = ('en', 'tgt')