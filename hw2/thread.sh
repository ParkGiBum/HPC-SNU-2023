#!/bin/bash

# Function to process each number
process_number() {
    number=$1
    ./run.sh -v -n 2 -t $number 1024 1024 1024
    # replace './your_command_here' with the command you want to run
}

export -f process_number

# Generate numbers from 1 to 256 and pass them to xargs for parallel execution
seq 1 256 | xargs -I {} -P 4 bash -c "process_number {}"

