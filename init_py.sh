#!/bin/bash

# Check if venv directory already exists
if [ -d "venv" ]; then
    echo "Virtual environment already exists. source sourceme to activate."
else
    echo "Creating new virtual environment..."
    python3 -m venv venv
fi

if [ -f "requirements.txt" ]; then
    echo "Installing requirements from requirements.txt..."
    source ./venv/bin/activate
    pip install -r requirements.txt
    deactivate
    echo "requirements installed. source sourceme to activate."
else
    echo "No requirements.txt found."
fi

