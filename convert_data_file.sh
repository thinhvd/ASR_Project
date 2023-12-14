#!/bin/bash

data_dir="./data"

mkdir -p "${data_dir}/train_clean_data"
mkdir -p "${data_dir}/test_clean_data"

# Extract train data
echo "Extracting train data..."
tar -xzf "${data_dir}/dev-clean.tar.gz" -C "${data_dir}/train_clean_data"

# Extract test data
echo "Extracting test data..."
tar -xzf "${data_dir}/test-clean.tar.gz" -C "${data_dir}/test_clean_data"

#-----------------------------------------------------------------#

# Convert files from flac to wav
echo "Converting .flac to .wav..."
find "${data_dir}" -name "*.flac" -exec sh -c 'sox "{}" "${0%.flac}.wav"' {} \;
#find "${data_dir}" -name "*.mp3" -exec sh -c 'sox "{}" "${0%.mp3}.wav"' {} \;
echo "Finished conversion."