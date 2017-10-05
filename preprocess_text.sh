#! /usr/bin/env bash

set -euo pipefail

infile=$1
outfile=${infile}.clean

# replace urls with <url>, map digits to zero, collapse whitespace, lowercase
sed -e "s#http : / / #http://#ig;s#https : / / #https://#ig;s/ ’ s / 's /ig;s/ ’ /'/g;s/-LRB-/(/g;s/-RRB-/)/g;s/http:[^\s]*/<url> /ig;s/www\.[^\s]*/<url> /ig;s/[0-9]/0/g" $f | tr -s "  " " " | tr "[:upper:]" "[:lower:]" < $infile > $outfile
