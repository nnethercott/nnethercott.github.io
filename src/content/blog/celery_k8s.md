---
title: "task queues in kubernetes ☸"
description: "over-engineering a flask service"
pubDate: "Jul 06 2024"
tags: ["k8s", "python"]
---

<!-- <div style="text-align: center;"> -->
<!--     <img src="https://cdn.analyticsvidhya.com/wp-content/uploads/2020/07/graphic4.jpg" style="width: 100%; display: block; margin: 0 auto;"> -->
<!-- </div> -->

I've recently been learning about kubernetes at work as my team handles the shift from distributed task queues to [argo jobs](https://argoproj.github.io/workflows/) for orchestrating steps in our machine learning pipeline. Before we fully nuke the old workflow I thought I'd write a post both as a reference for myself and hopefully as a guide for other ML engineers new to the ops-side of things on how to implement a scaleable distributed task queue using [Celery](https://docs.celeryq.dev/en/stable/getting-started/introduction.html), [rabbitmq](https://www.rabbitmq.com/), and kubernetes.

Let's pretend we have a basic service like the one below. (flask app with long job and short job).


mention this repo: https://github.com/nnethercott/celery-flask-k8s
