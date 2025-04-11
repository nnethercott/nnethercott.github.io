---
title: "Improving model encryption on mobile"
description: "Avoiding OOMs in production"
pubDate: "April 1 2025"
tags: ["dart", "flutter"]
---
reference on phrasing of describing encryption in english [here](https://jdrouet.github.io/posts/202503170800-search-engine-part-1/) 

- initial implementation flame chart, only knew when issues showed up after integrating the plugin 
- revisted the flame chart and saw ~this
- a few observations; it looks like there's a memory leak (note the steadily increasing RAM usage=>GC is not happening (could it be intermediate buffers => validate this with the debugger), we have no control of the internals here, we'll have to re-implement it)
  - a first observation when reading the docs on a method (always read the docs kids!)
- we'd probably aim for a streaming approach
  - this works but still have a memory leak :(

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

