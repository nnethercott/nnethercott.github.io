---
title: "Chasing down a memory leak"
description: "garbage collection is fun"
pubDate: "April 1 2025"
tags: ["dart"]
---

points:
- comparing the first implementation to various iterates
- self-time in the call tree, using dart dev tools to look for memory leak 
  - talk about pre-allocating buffer to remove time spent in memory allocation of dynamic list 
- some primer on how GC works in dart
  - maybe an attempt at Future.delayed(Duration.zero) everywhere 
  - read some gh issues/threads on cleanup in isolates
- compare how code runs in isolate vs outside from perfs perspective
- switching out the encryption backends ?
- dart dev tools collects cpu times incorrectly [link](https://github.com/flutter/devtools/issues/8870) 

