---
layout: page
---


  {% for post in site.posts %}
      [{{ post.title }}]({{ post.url }})
      {{ post.excerpt }}
    {% if post.excerpt != post.content %}
            [Read More...]({{site.baseurl}}{{post.url}})
    {% endif %}
    
  {% endfor %}

