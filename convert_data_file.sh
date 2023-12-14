#!/bin/bash

data_dir="./data"

mkdir -p "${data_dir}/train_clean_data"
mkdir -p "${data_dir}/test_clean_data"
mkdir -p "${data_dir}/train_other_data"
mkdir -p "${data_dir}/test_other_data"

# Function to extract data
extract_data() {
  archive=$1
  destination=$2
  echo "Extracting data from $archive to $destination..."
  if tar -xzf "$archive" -C "$destination" &; then
    echo "Extraction started successfully."
  else
    echo "Extraction failed. Exiting..."
    exit 1
  fi
}

# Extract train clean data
extract_data "${data_dir}/dev-clean.tar.gz" "${data_dir}/train_clean_data"

# Extract test clean data
extract_data "${data_dir}/test-clean.tar.gz" "${data_dir}/test_clean_data"

# Extract train other data
extract_data "${data_dir}/dev-other.tar.gz" "${data_dir}/train_other_data"

# Extract test other data
extract_data "${data_dir}/test-other.tar.gz" "${data_dir}/test_other_data"

# Wait for all background processes to finish
wait

echo "All extractions completed successfully."


#-----------------------------------------------------------------#

# Convert files from flac to wav
echo "Converting .flac to .wav..."
find "${data_dir}" -name "*.flac" -exec sh -c 'sox "{}" "${0%.flac}.wav"' {} \;
#find "${data_dir}" -name "*.mp3" -exec sh -c 'sox "{}" "${0%.mp3}.wav"' {} \;
echo "Finished conversion."