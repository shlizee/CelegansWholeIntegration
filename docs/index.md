---
layout: page
type: homepage
---


{% for post in site.posts %}
    <div>
      <a href="{{ post.url }}">{{ post.title }}</a>
      {{ post.excerpt }}
      {% if post.excerpt != post.content %}
        <a href="{{ site.baseurl }}{{ post.url }}">Read More ...</a>
      {% endif %}
    </div>
{% endfor %}



