---
title: "Interior mutability design pattern"
description: "Arc<Mutex<CoolArticle>>"
pubDate: "Oct 10 2024"
tags: ["rust"]
---

I first got introduced to this design pattern in the [Rust book](https://doc.rust-lang.org/book/ch16-00-concurrency.html) and later in the [async rust book](https://rust-lang.github.io/async-book/). What I like about this pattern is that it encourages you to reflect more on the API you're designing in terms of private/public access for objects. In some cases you don't want users to be able to use an object in a mutable way but you still might need to modify the internals of the instance during its lifetime. Interior mutability offers a solution to that. 

If that sounded confusing, don't worry, I'll clear things up as we go on.

According to the book, *interior mutability* is a "design pattern in Rust that allows you to mutate data even when there are immutable references to that data". What this essentially means is that you can write code like this:

```rust 
use std::cell::RefCell;

fn main() {
    let obj = RefCell::new(1u32); // obj isn't mut
    *obj.borrow_mut() += 1;
    assert_eq!(obj.into_inner(), 2);
}
```
The important line here is the `*obj.borrow_mut()+=1` part. Here we're modifying the internals of the RefCell even though the variable `foo` wasn't declared as mut. 

`RefCell` is a "mutable memory location with dynamically checked borrow rules". It's a smart pointer that gives you the option to modify the wrapped value even if the pointer itself isn't declared `mut`. The dynamically checked borrow rules part means Rust's borrowing rules are enforced at *runtime* instead of *compile time* - code written using RefCell's will panic instead of refusing to compile. The only downside to RefCells is that they're not thread-safe meaning we can't move the data to a new thread. 

If you want mutable access to an object in a multi-threaded context the `Arc<Mutex<T>>` pattern instead comes in handy. 

Specifying a bit; 
* `Arc` is a thread-safe reference counted pointer which is Send
* `Mutex` ensure exclusive acess to the shared data by a single thread at any given time

To see the pattern in action checkout the snippet below. Here we spawn two worker threads which each increment the value of some shared underlying data. 

```rust 
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

fn main() {
    let data: Arc<Mutex<i32>> = Arc::new(Mutex::new(0));
    let mut handles: Vec<JoinHandle<()>> = vec![];

    for _ in 0..2 {
        let data_ = Arc::clone(&data);
        handles.push(thread::spawn(move || {
            let mut d = data_.lock().unwrap();
            *d += 1  // increment data in each thread
        }));
    }

    // make sure main thread waits for workers to finish
    for handle in handles {
        handle.join().unwrap();
    }

    println!("{:?}", data.lock().unwrap()); //prints 2
}
```

Ok so why is this useful ?

I'll admit it's hard to come up with non trivial use cases off the top of my head for this pattern. If you're new to rust it can look a bit daunting, and it's hard to imagine a situation where you'd need multiple threads each having mutable access to some object. Instead of inventing some abstract example here, I thought I'd motivate the pattern by showing how I used it recently in a [multi-threaded async image downloader](https://github.com/nnethercott/tiny-data).


The problem: We have several urls we want to download in multiple threads and we wish to give users a visual cue to notify them on the job progress. 

The code structure looks like this:

```rust 
//download task 
pub struct Task {
    pub downloader: Arc<Downloader>,
    pub url: impl IntoUrl,
}

// downloader 
pub struct Downloader {
    pub progress_bar: Mutex<ProgressBar>, // shared progress bar 
}

impl Downloader{
    pub async fn download( 
            &self,
            url: impl IntoUrl,
            filename: String,
        ) -> Result<u8, Box<dyn std::error::Error>> {
        todo!()
    }
}

// task scheduler 
pub struct DLManager {
    pub target_size: usize,
    pub downloader: Arc<Downloader>,
}
```

The `DLManager`s job is to create a download `Task` out of user-provided url which gets executed asynchronously by the manager's `Downloader`. Pretty simple stuff (except for the async part, that can get a bit confusing).

Aside: for a review on async concepts I highly recommend the [Asynchronous Programming in Rust book](https://rust-lang.github.io/async-book/), as well as [Tokio's docs](https://tokio.rs/tokio/tutorial/async). 

Notice in the code block above how we have `Arc` and `Mutex` present. The Arc's should tip you off that the Tasks are going to be sent to worker threads for execution, and the Mutex on the Downloader means we're planning on modifying the object's internals at some point. Specifically, we want to increment the Downloader's progress bar by 1 every time a task successfully finishes and display that to users.

In action that looks like:
<figure>
<div style="text-align: center;">
    <img src="https://github.com/nnethercott/tiny-data/raw/main/assets/images/demo.gif?raw=true" style="width: 100%; display: block; margin: 0 auto;" >
      <figcaption></figcaption>
</div>
</figure>

To acheive this beautiful multi-threaded progress bar the `Downloader.download` method I mentioned earlier looks like this: 

```rust 
impl Downloader {
    pub async fn download(
        &self,
        url: impl IntoUrl,
        filename: String,
    ) -> Result<u8, Box<dyn std::error::Error>> {
        let res = reqwest::get(url).await?;
        match res.status() {
            StatusCode::OK => {
                let bytes = res.bytes().await?;

                if !bytes.starts_with(b"<!DOCTYPE html>") {
                    let mut file = AsyncFile::create(filename).await?;
                    file.write_all(&bytes).await?;

                    //interior mutability + thread-safe access !
                    {
                        self.progress_bar.lock().unwrap().inc(1);
                    }
                }
            }
            _ => return Ok(0),
        }

        Ok(1)
    }
}
```
Although its only a one-liner here -- *self.progress_bar.lock().unwrap().inc(1)* is playing a critical role under the hood. Whenever a downloading task completes within a worker thread we're acquiring exclusive access to the shared progress bar and safely incrementing its counter. With this structure, users **don't** need to declare their Downloader as mutable in order to benefit from the multi-progress bar functionality. And why should they? A struct with one utlity method like the Downloader above generally isn't considered as a stateful entity for intents and purposes. 

<!-- The reason I chose to structure my code like this is since it doesn't make sense for users to have to declare their `Downloader` as mut in order to see the job progress. The Downloader struct is just a util meant to execute a basic url get, from the user-side *it doesn't need to be stateful*. --> 

In retrospect it may have been cleaner to decouple the Downloader from the ProgressBar since they're not inherently linked to one another, but at the same time I think the Arc<Downloader<Mutex\<ProgressBar>>> layout is a fun one. 

Taking a step back, this pattern lets us:
* keep track of the download jobs and communicate progress in real time to users
* spawn new download tasks in a multithreaded environment
* does all of that without needing mutability on the client-side

That's it for this one!
