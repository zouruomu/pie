#!/bin/bash
# usage: after.sh <file-to-wait-for> <grep pattern> -- command...  : runs command once the pattern is in the file
f=$1; pat=$2; shift 3
until grep -qE "$pat" $f 2>/dev/null; do sleep 10; done
exec "$@"
