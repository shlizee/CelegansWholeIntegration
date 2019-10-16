---
layout: page
type: homepage
---

<ul>
  {% for post in site.posts %}
    <li>
      <h2><a href="{{ post.url }}">{{ post.title }}</a></h2>
      <p>{{ post.excerpt }}</p>
      {% if post.excerpt != post.content %}
        <h3><a href="{{ site.baseurl }}{{ post.url }}">Read More ...</a></h3>
      {% endif %}
    </li>
  {% endfor %}
</ul>
