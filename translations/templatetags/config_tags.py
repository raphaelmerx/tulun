from django import template
from translations.models import SystemConfiguration

register = template.Library()

@register.simple_tag
def get_config():
    """
    Returns the system configuration instance for use in templates
    """
    return SystemConfiguration.load()