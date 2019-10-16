---
layout: page
type: homepage
---

<ul>
  {% for post in site.posts %}
    <li>
      <h3><a href="{{ site.baseurl }}{{ post.url }}">{{ post.title }}</a></h3>
      <p>{{ post.excerpt }}</p>
        {% if post.excerpt != post.content %}
          <h4><a href="{{ site.baseurl }}{{ post.url }}">Read More ...</a></h4>
        {% endif %}
    </li>
  {% endfor %}
</ul>
