#!/bin/bash

# dev-clean: https://www.openslr.org/resources/12/dev-clean.tar.gz
# dev-other: https://www.openslr.org/resources/12/dev-other.tar.gz
# test-clean: https://www.openslr.org/resources/12/test-clean.tar.gz
# test-other: https://www.openslr.org/resources/12/test-other.tar.gz

# if [ "$#" -ne 2 ]; then
#   echo "Usage: $0 <train_data_url> <test_data_url>"
#   exit 1
# fi

train_data_url="https://www.openslr.org/resources/12/dev-clean.tar.gz"
test_data_url="https://www.openslr.org/resources/12/test-clean.tar.gz"
data_dir="./data"

# Function to check if URL is valid
check_url() {
  url=$1
  if curl --output /dev/null --silent --head --fail "$url"; then
    echo "URL $url is valid."
  else
    echo "URL $url is not valid or unreachable."
    exit 1
  fi
}

# Function to download data
download_data() {
  url=$1
  echo "Downloading data from $url..."
  if wget -O "${data_dir}/$(basename $url)" "$url"; then
    echo "Download completed successfully."
  else
    echo "Download failed. Exiting..."
    exit 1
  fi
}

# Check URLs
check_url "$train_data_url"
check_url "$test_data_url"

# Create data directory if it doesn't exist
mkdir -p "$data_dir"

# Download train data
download_data "$train_data_url"

# Download test data
download_data "$test_data_url"
