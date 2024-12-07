#!/bin/bash

# Путь к каталогу, за которым нужно следить
DIRECTORY="/home/qwest/project/PycharmProjects/Reinforsment_Learning/VAE/weights/main/"

inotifywait -m "$DIRECTORY" -e create -e moved_to |
while read -r event file; do
  if [[ -f "$file" ]]; then
    echo "New file detected: $file"
  fi
done