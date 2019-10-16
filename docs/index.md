---
layout: page
type: homepage
---


{% for post in site.posts %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a>
      {{ post.excerpt }}
      
      {% if post.excerpt != post.content %}
        <a href="{{ site.baseurl }}{{ post.url }}">Read More ...</a>
      {% endif %}
    </li>
{% endfor %}



