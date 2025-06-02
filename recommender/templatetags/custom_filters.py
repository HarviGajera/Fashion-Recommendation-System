from django import template
register = template.Library()

@register.filter
def render_stars(rating):
    try:
        rating = float(rating)
    except (TypeError, ValueError):
        rating = 0

    full_stars = int(rating)
    half_star = 1 if rating - full_stars >= 0.5 else 0
    empty_stars = 5 - full_stars - half_star

    html = ''
    html += '<i class="fas fa-star"></i>' * full_stars
    html += '<i class="fas fa-star-half-alt"></i>' * half_star
    html += '<i class="far fa-star"></i>' * empty_stars
    html += f' <span class="rating-number">({rating:.1f})</span>'
    return html