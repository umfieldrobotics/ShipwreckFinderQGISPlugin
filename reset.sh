#!/bin/bash

FILE_PATH="/home/smitd/DrewShipwreckSeeker/ShipwreckSeekerQGISPlugin/dist/python-package-1.0.1-qgis.zip"

SCRIPT="/home/smitd/DrewShipwreckSeeker/ShipwreckSeekerQGISPlugin/create_plugin.py"

if [ -f "$FILE_PATH" ]; then
	echo "Deleting $FILE_PATH"
	rm "$FILE_PATH"
else
	echo "File $FILE_PATH does not exist"
fi

echo "Running pything script to create plugin"

python3 "$SCRIPT"

echo "Done"
