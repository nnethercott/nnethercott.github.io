---
title: "Improving model encryption on mobile"
description: "Some optimizations made for more performant cryptography operations"
pubDate: "April 1 2025"
tags: ["dart", "flutter"]
---

points:
- self-time in the call tree, using dart dev tools to look for memory leak 
  - talk about pre-allocating buffer to remove time spent in memory allocation of dynamic list 
- some primer on how GC works in dart
  - maybe an attempt at Future.delayed(Duration.zero) everywhere 
  - read some gh issues/threads on cleanup in isolates
- compare how code runs in isolate vs outside from perfs perspective
- switching out the encryption backends ?

notes:
- isolates don't share memory with the main process [link](https://docs.flutter.dev/packages-and-plugins/background-processes) 
- flutter/dart dev tools cpu profiler times are off 

