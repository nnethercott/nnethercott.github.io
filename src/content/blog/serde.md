---
title: "serde, serde_json, and serde_with"
description: "A quick look at the serde ecosystem; macros syntax for convenient serialization in rust"
pubDate: "March 6 2025"
tags: ["rust"]
---

At first I didn't really understand the hype around [serde](https://serde.rs/) (*ser*ialize-*de*serialize). Coming from a Python background, a lot of the complexity in serializing objects is abstracted from you using frameworks like Django or Pydantic. Recently though, I came across a use case during a refactor on my [tokenizer project](https://github.com/nnethercott/toktkn) that motivated its novelty for me.
<div style="text-align: center;">
    <img src="/media/serde_post/graphic.png" style="width: 50%; display: block; margin: 0 auto;">
</div>

The specific problem I was dealing with was reading and writing tokenizers from disk after training them on a dataset. Depending on the size of the corpus, training can time-intensive process and we'd like to cache the work after we're done to reload later or share with others. 

At its core, a tokenizer is basically a wrapper around a HashMap storing rules for merging substrings, and because of this they can have memory footprints that are non-trivially large. For example, gpt2 with a vocabulary size of only ~50k needs a tokenizer taking up nearly 1.4 MB -- today models have vocabularies nearly 3x that. When serializing a tokenizer we need to consider this memory cost and only keep what's absolutely necessary for deserialization.

Consider the code below outlining the interface of a tokenizer -- if it seems confusing, don't worry we'll go through it together.
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

Serde is a crate in Rust that provides an interface based on procedural macros (*proc-macros*) to define serialization behaviour, with the traits `Serialize` and `Deserialize` implementing the necessary io operations under the hood. An important thing to note is that structs deriving these traits must have fields which themselves **also** derive Serialize/Deserialize, e.g. you can't do something like: 

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

While serde supports HashMaps with heterogeneous key-value pairs -- including (u32,u32) tuples like the BPETokenizer.encoder -- [serde-json](https://github.com/serde-rs/json) does not. Serializing to valid json restricts us to maps with **string keys only**. That's why the `special_tokens_map` of the `TokenizerConfig` is already good as is. We could instead use another format like CBOR and it would work as is, but we lose out on readability (and I wouldn't have content for my blog).

In Python we could normally do a quick: 

```python
encoder = {str(k): v for k,v in encoder.items()}
```

and call it a day - but here we're working in Rust !

Rather than manually writing the code to encode and decode (u32, u32) tuples from strings though, we can make use of the [serde-with](https://docs.rs/serde_with/latest/serde_with/index.html) crate. Serde-with provides its own set of macros to allow for more custom serialization/deserialization. 

Let's take a minute to appreciate the [DisplayFromStr](https://docs.rs/serde_with/3.12.0/serde_with/struct.DisplayFromStr.html) struct from serde-with that I used in the code snippet at the very start. 

DisplayFromStr relies on your struct implementing both the Display and FromStr traits, and maps to and from the string representation you defined. Taking a look at the [source code](https://docs.rs/serde_with/3.12.0/src/serde_with/ser/impls.rs.html#538-548) we see


```rust
impl<T> SerializeAs<T> for DisplayFromStr
where
    T: Display,
{
    fn serialize_as<S>(source: &T, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.collect_str(source)
    }
}
```
Now following that `collect_str` method for the Serialize trait we get that DisplayFromStr relies on your struct's implementation of `to_string`:
```rust
// trait Serialize
 fn collect_str<T>(self, value: &T) -> Result<Self::Ok, Self::Error>
    where
        T: ?Sized + Display,
    {
        self.serialize_str(&value.to_string())
    }
```
Since u32 implements Display and FromStr we can annotate the field in our struct by 

```rust
#[serde_as]
#[derive(Serialize, Deserialize)]
pub struct BPETokenizer {
    #[serde_as(as = "Vec<((DisplayFromStr, DisplayFromStr), DisplayFromStr)>")]
    pub encoder: HashMap<u32, <u32,u32>>,

    // rest of fields 
}
```

and we get se/deserialization for free! 
