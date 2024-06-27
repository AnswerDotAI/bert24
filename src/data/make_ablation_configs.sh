#!/bin/bash

for config_yaml in configs/relative/*.yaml; do
    echo $config_yaml
    python relative_prop_to_instance_prop.py --config $config_yaml
done
