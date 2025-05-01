---
title: 'On the subject of Cargo workspaces'
description: 'Why disabling default features is ignored in Cargo workspaces'
pubDate: 'May 1 2025'
tags: ["rust", "cargo"]
---
In a Cargo workspace packages share a common Cargo.lock and /target, this way all your crates compile against the same version of a given dependency. 

Workspaces are great for a few reasons; you can split up a big package into several subcomponents to avoid triggering re-compilation when you make small changes in one part of your code, and you ensure consistency in dependency versioning between your packages. At a high level they also make your project more readable.

Cargo features, on the other hand, allow for conditional behaviour of your package through the `#[cfg]` attribute at compile time and `cfg!()` macro at runtime depending on which features are enabled.

Cargo workspaces and Cargo features by themselves are great tools for structuring your package, but used together they can lead to some unexpected behaviour if you're not careful, especially when enabling a default feature in one of your packages. 

Suppose we have a workspace layout like this :

```bash
some_crate
├── Cargo.lock
├── Cargo.toml
├── example
│   ├── Cargo.toml
│   ├── src
│   └── target
└── crate_with_optional_features
    ├── Cargo.toml
    ├── src
    └── target
```

and that `crate_with_optional_features` defines a few optional features; 
```toml
# crate_with_optional_features/Cargo.toml
[package]
name = "crate_with_optional_features"
version = "0.1.0"
edition = "2024"

[lib]
path = "./src/lib.rs"

[dependencies]
serde = { version = "1.0.219", features = [ "derive" ], optional=true }
serde_json = { version = "1.0.140", optional=true }

[dependencies.tokio]
version = "1.44.2"
features = ["rt-multi-thread", "macros", "time"]

[features]
# default features that get compiled automatically
default = ["derive"]
# an optional feature enabling serialization with serde
derive = ["serde", "serde_json"]
```
In this dummy example we have a package that depends on tokio, and in which theres **two** features; "derive" which enables serde and serde_json, and the special "default" feature which enables the derive feature.

Finally, suppose in `crate_with_optional_features/src/lib.rs` we use the `#[cfg(feature = "derive")]` attribute to define a module and some associated code that gets compiled if "derive" is active.

```rust 
use std::time::Duration;

#[cfg(feature = "derive")]
pub mod bar {
    use serde::Serialize;

    #[derive(Serialize)]
    struct Foo {
        inner: i32,
    }

    pub fn do_some_stuff() {
        let f = Foo { inner: 42 };
        println!("{}", serde_json::to_string(&f).unwrap());
    }
}

pub fn tokio_do_something() -> std::io::Result<()> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        tokio::time::sleep(Duration::from_secs(1)).await;
        println!("done");
    });

    // if derive enabled, run this
    #[cfg(feature = "derive")]
    bar::do_some_stuff();

    Ok(())
}
```

**Here's where it get's weird:** let's use `crate_with_optional_features` in `example` but disable the default features. 
```toml
# example/Cargo.toml
[package]
name = "example"
version = "0.1.0"
edition = "2024"

[[bin]]
name = "ex"
path = "./src/main.rs"

[dependencies.crate_with_optional_features]
path = "../crate_with_optional_features/"
default-features = false # DON'T enable serde
```
and let the `example/main.rs` look like:

```rust
use crate_with_optional_features::tokio_do_something;

fn main() {
    // we disabled `default` this should sleep for 2 seconds and print "done"
    tokio_do_something();
}
```

What you actually get when you run this is the following (note how "{inner: 42}" is still printed even though we disabled the derive feature):
```bash
~/co/rus/p/optional_features_in_crate ❯ cargo run --bin ex
done
{"inner":42}
```

Why is this ? Since both packages share a single Cargo.lock and by default we've enabled the derive feature in `crate_with_optional_features`, serde and serde_json are compiled and show up in the lockfile. Because of this we essentially ignore the `default-features = false` declaration in example's Cargo.toml and compile it anyways. We can verify this by running a cargo tree:

```bash
~/co/rus/p/optional_features_in_crate ❯ cargo tree
example v0.1.0 
└── crate_with_optional_features v0.1.0
    ├── serde v1.0.219
    │   └── serde_derive v1.0.219 (proc-macro)
    │       ├── proc-macro2 v1.0.95
    │       │   └── unicode-ident v1.0.18
    │       ├── quote v1.0.40
    │       │   └── proc-macro2 v1.0.95 (*)
    │       └── syn v2.0.101
    │           ├── proc-macro2 v1.0.95 (*)
    │           ├── quote v1.0.40 (*)
    │           └── unicode-ident v1.0.18
    ├── serde_json v1.0.140
    │   ├── itoa v1.0.15
    │   ├── memchr v2.7.4
    │   ├── ryu v1.0.20
    │   └── serde v1.0.219 (*)
    └── tokio v1.44.2
        ├── pin-project-lite v0.2.16
        └── tokio-macros v2.5.0 (proc-macro)
            ├── proc-macro2 v1.0.95 (*)
            ├── quote v1.0.40 (*)
            └── syn v2.0.101 (*)
```

*So even though we disabled the derive feature we're still compiling serde and serde_json*!

How can we fix this ? 

The easiest approach is to just remove the default feature from `crate_with_optional_features`, this way both serde deps are optional and Cargo won't fetch them when building the workspace packages. Doing this our cargo tree becomes: 

```bash
~/co/rus/p/optional_features_in_crate ❯ cargo tree
example v0.1.0
└── optional_features v0.1.0
    └── tokio v1.44.2
        ├── pin-project-lite v0.2.16
        └── tokio-macros v2.5.0 (proc-macro)
            ├── proc-macro2 v1.0.95
            │   └── unicode-ident v1.0.18
            ├── quote v1.0.40
            │   └── proc-macro2 v1.0.95 (*)
            └── syn v2.0.101
                ├── proc-macro2 v1.0.95 (*)
                ├── quote v1.0.40 (*)
                └── unicode-ident v1.0.18
```
And the code works again 
```bash
~/co/rus/p/optional_features_in_crate ❯ cargo run --bin ex
done
```
Nice !

*But what if I want to keep the derive feature enabled by default?*

Well in that case you need to move away from the Cargo workspaces structure so that each package gets its own Cargo.lock. Doing this, the new structure becomes as follows:
```bash
some_crate
├── example
│   ├── Cargo.lock
│   ├── Cargo.toml
│   ├── src
│   └── target
└── crate_with_optional_features
    ├── Cargo.lock
    ├── Cargo.toml
    ├── src
    └── target
```

And the code also works again !
```bash
~/co/rus/p/optional_features_in_crate ❯ cargo run --bin ex
done
```

## References
* [The book](https://doc.rust-lang.org/book/ch14-03-cargo-workspaces.html)
* [The Cargo book](https://doc.rust-lang.org/cargo/reference/features.html)
