#!/bin/bash
if [ -d "datasets/" ]; then rm -Rf "datasets/"; fi
wget https://sound-field-db.s3.amazonaws.com/sample_datasets.tgz
tar zxvf sample_datasets.tgz
rm sample_datasets.tgz
mv sample_datasets/ datasets/
