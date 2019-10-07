<ul>
{% for post in site.posts %}
  <li>
    <a href="{% post.permalink %}">{% post.title %}</a>
    <p>{{ post.excerpt }}</p>
  </li>
{% endfor %}
</ul>
