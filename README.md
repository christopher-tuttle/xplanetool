X-Plane Tool README
===================

This is part of a side project, and it is published because folk were asking
about interpreting the raster layers in X-Plane 12's DSF files.

Building
--------

 1. Install Rust: https://rustup.rs
 1. `cargo build --release`

The only unusual dependency is `compress-tools`, which is used to support 7z-compressed
files. If using homebrew on mac, this requires `libarchive`, which gets installed in a
path that `pkg-config` won't necessarily find. After `brew install libarchive`, I added
this to my `.bashrc`:
`export PKG_CONFIG_PATH="$PKG_CONFIG_PATH:/usr/local/Cellar/libarchive/3.6.1/lib/pkgconfig/"`

Running
-------
To make some funny looking pngs from the rasters in a DSF:

`cargo run --release -- png /Users/clt/Desktop/flight/X-Plane\ 12/Global\ Scenery/X-Plane\ 12\ Global\ Scenery/Earth\ nav\ data/+10-160/+19-156.dsf`

To convert a DSF to text -- just like the `dsf2text` mode of `dsftool`:

`cargo run --release -- dsf2text <input> <output>`

Use `--vmodule=trace` or `--vmodule=debug` to add a lot of verbosity.

Bugs
----
Yup, probably many. Notably the path-fu might be sad on Windows; I'm using a mac.

