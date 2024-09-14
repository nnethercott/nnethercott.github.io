---
title: "Interior mutability design pattern in rust"
description: "Arc<Mutex<CoolArticle>>"
pubDate: "Jun 16 2024"
tags: ["rust"]
---

comments on how we want to keep the object immutable (we don't want users modifying it) but we would still like to modify some fields under the hood.

talk about how that works for my downloading progress bar

<figure>
<div style="text-align: center;">
    <img src="https://github.com/nnethercott/tiny-data/raw/main/assets/images/demo.gif?raw=true" style="width: 100%; display: block; margin: 0 auto;" >
      <figcaption></figcaption>
</div>
</figure>
