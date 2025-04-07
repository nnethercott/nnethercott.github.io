---
title: "Useful serde"
description: "Some macros syntax for convenient serialization in rust 🥣"
pubDate: "March 6 2025"
tags: ["rust"]
---
<div style="text-align: center;">
    <img src="/media/serde_post/graphic.png" style="width: 50%; display: block; margin: 0 auto;">
</div>

At first I didn't really understand the hype around [serde](https://serde.rs/) (*ser*ialize-*de*serialize). Coming from a Python background, a lot of the complexity in serializing objects is abstracted from you using frameworks like Django or Pydantic. Recently though, I came across a use case during a recent refactor on my [tokenizer project](https://github.com/nnethercott/toktkn) that really motivated its novelty for me.

Consider the code below -- if it seems confusing, don't worry we'll go through it together.

```rust 
// ... 
use serde::{Deserialize, Serialize};
use serde_with::{serde_as, DisplayFromStr};

#[derive(Serialize, Deserialize, Clone)]
pub struct TokenizerConfig {
    // number of words in tokenizer
    pub vocab_size: usize,
    // custom character sequences to tokenize
    pub special_tokens_map: Option<HashMap<String, Token>>,
    // preprocessing enum 
    #[serde(default)]
    pub preproc: Normalizer,
}

// BPE tokenizer derived from the config
#[serde_as]
#[derive(Serialize, Deserialize)]
pub struct BPETokenizer {
    pub config: TokenizerConfig,
    #[serde_as(as = "Vec<((DisplayFromStr, DisplayFromStr), DisplayFromStr)>")]
    pub encoder: HashMap<u32, <u32,u32>>,
    #[serde(skip)]
    pub decoder: HashMap<<u32,u32>,u32>,
}
```

Serde provides an interface based on procedural macros (*proc-macros*) to define serialization behaviour, with the traits `Serialize` and `Deserialize` implementing the necessary io operations under the hood.

An important thing to note is that structs deriving these traits must have fields which themselves **also** derive Serialize/Deserialize, e.g. you can't do something like: 

```rust 
use serde::Serialize; 

struct Foo; // <-- doesn't implement serde::Serialize!

#[derive(Serialize)]
struct Bar(Foo)
```

Since I was rolling my own version of the [tokenizers](https://docs.rs/tokenizers/latest/tokenizers/) crate, I wanted to emulate the ability to save and load pretrained tokenizers from disk. Specifically, I wanted the following behaviour:
1. the combination rules for the tokenizer (e.g. the *encoder* of the BPETokenizer) should be serialized to a HashMap<<u32, u32>, u32> -- obvious right ?
2. we can save space by skipping the serialization of the token expansion rules (the inverse HashMap<u32, <u32,u32>> of the bullet above)
3. when deserializing, if no preprocessing strategy is defined in the .json spec then use a default normalizing strategy 


Reading the [serde docs](https://serde.rs/field-attrs.html) we immediately find a solution to our 2nd and 3rd points above in the form of **#[serde(skip)]** and **#[serde(default)]**. The docs are pretty straight forward in describing them; 

> #[serde(skip)]<br>Skip this field: do not serialize or deserialize it.

and 

> #[serde(default)]<br>If the value is not present when deserializing, use the Default::default().

An important caveat about the last one; in order for **#[serde(default)]** to work the field needs to implement the [default trait](https://doc.rust-lang.org/std/default/trait.Default.html) !
