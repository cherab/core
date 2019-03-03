#!/bin/bash

echo Removing all .c, .so and .html files...

find cherab -type f -name '*.c' -exec rm {} +
find cherab -type f -name '*.so' -exec rm {} +
find cherab -type f -name '*.html' -exec rm {} +
rm build -rf

