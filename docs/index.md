---
layout: page
---


  {% for post in site.posts %}
      #<a href="{{ post.url }}">{{ post.title }}</a>
      {{ post.excerpt }}
      
    {% if post.excerpt != post.content %}
      #<a href="{{ site.baseurl }}{{ post.url }}">Read more</a>
    {% endif %}
    
  {% endfor %}

