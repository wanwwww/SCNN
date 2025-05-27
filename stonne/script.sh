#!/bin/bash

for topo in ./topo_files/*.csv
do
    name=$(basename "$topo" .csv)
    echo "Start running $name..."
    ./stonne "$topo" ./cfg_files/tpu_4_4.cfg > "./output_0/${name}_output.txt" &
done

wait

echo "All topology tests are completed!"