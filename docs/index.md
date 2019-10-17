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

This blog is part of [_CelegansWholeIntegration_ repository](https://github.com/shlizee/CelegansWholeIntegration). We are preparing to release the code and will notify about the release in this blog. Also refer to _CelegansWholeIntegration_ preprint for additional details about the work: [BiorXiv Preprint](https://www.biorxiv.org/content/10.1101/724328v1).

