# Sagacity
A simple library for building, training and using neural networks.

The code compiles on Linux (haven't tryed it on Windows).
The project only contains some very basic tests and needs better ones.

Future goals:
  - Separate the training algorithm from the network
  - Maybe make bindings for Python and Java

In order to build this project you'll need to have the following packages:
  - build-essential
  - cmake
  - nlohmann-json3-dev

The nlohmann json library is used because the standard Ubuntu repository doesn't yet host the boost equivalent.

If you also want to build the tests you'll need to install:
  - libboost-test-dev