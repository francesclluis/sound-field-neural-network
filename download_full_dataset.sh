#!/bin/bash
if [ -d "datasets/" ]; then rm -Rf "datasets/"; fi
wget https://sound-field-db.s3.amazonaws.com/datasets.tgz
tar zxvf datasets.tgz
rm datasets.tgz
