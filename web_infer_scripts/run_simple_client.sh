#!/usr/bin/bash
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# From Genie-Envisioner (AgibotTech). Non-commercial use only.

IP_ADDRESS_OF_SERVER="localhost"

python3 web_infer_scripts/simple_client.py --host $IP_ADDRESS_OF_SERVER --port 8001 --env WM