#!/usr/bin/env python3

# This script scrapes data for Sock-shop using Prometheus

import argparse

import requests
import numpy as np
import pandas as pd

HOST_TEMPLATE = "http://{}:31090/"
MAX_RESOLUTION = 11_000 # Maximum resolution of Prometheus
SOCK_SHOP_NS = "sock-shop" # # Sock-shop namespace

# Names of the sock-shop containers
# The data will be collected for all of these containers
CONTAINERS = {
    'carts',
    'carts-db',
    'catalogue',
    'catalogue-db',
    'front-end',
    'orders',
    'orders-db',
    'payment',
    'queue-master',
    'rabbitmq',
    'shipping',
    'user',
    'user-db'
}

STEP = 1 # In seconds
DURATION = "45s"
QUERIES = {
            "cpu": f"sum(rate(container_cpu_usage_seconds_total{{namespace='{SOCK_SHOP_NS}'}} [{DURATION}])) by (container) * 100",
            "mem": f"sum(container_memory_usage_bytes{{namespace='{SOCK_SHOP_NS}'}}) by (container)",
            "lod": f"sum(rate(request_duration_seconds_count [{DURATION}])) by (name)",
            "lat_50": f"histogram_quantile(0.50, sum(rate(request_duration_seconds_bucket{{kubernetes_namespace=~'{SOCK_SHOP_NS}'}} [{DURATION}])) by (name, le))",
            "lat_90": f"histogram_quantile(0.90, sum(rate(request_duration_seconds_bucket{{kubernetes_namespace=~'{SOCK_SHOP_NS}'}} [{DURATION}])) by (name, le))",
            "lat_99": f"histogram_quantile(0.99, sum(rate(request_duration_seconds_bucket{{kubernetes_namespace=~'{SOCK_SHOP_NS}'}} [{DURATION}])) by (name, le))",
            "err": f"sum(rate(request_duration_seconds_count{{kubernetes_namespace='{SOCK_SHOP_NS}', status_code=~'4.+|5.+'}} [{DURATION}])) by (name) / sum(rate(request_duration_seconds_count{{kubernetes_namespace='{SOCK_SHOP_NS}'}} [{DURATION}])) by (name)"
}

# Merge two dictionaries of lists by appending the entries to the list.
# y will be append at the end of x
def _merge(x, y):
    data = x
    for key in y:
        data[key] = x.get(key, []) + y[key]
    return data

def _exec_query(query, start_time, end_time, host):
    response = requests.get(host + '/api/v1/query_range',
                            params={
                                'query' : query,
                                'start' : start_time,
                                'end'   : end_time,
                                'step'  : f"{STEP}s"
                                })
    data = {}
    results = response.json()['data']['result']
    for result in results:
        if all(k not in result['metric'].keys() for k in ['container', 'name']): continue
        if 'container' in result['metric']:
            service_name = result['metric']['container']
        else:
            service_name = result['metric']['name']
        if service_name not in CONTAINERS: continue
        data[service_name] = result['values']
    return data

# Given a valid query, extracts the relevant data
def exec_query(query, start, end, host):
    # If all the data can be collected in only one request
    if not (end - start) / STEP > MAX_RESOLUTION:
        return _exec_query(query, start, end, host)

    data = {}
    start_time = start
    end_time = start
    while end_time < end:
        end_time = min(end_time + MAX_RESOLUTION, end)
        print(f"Querying data from {start_time} to {end_time}")
        d = _exec_query(query, start_time, end_time, host)
        data = _merge(data, d)
        start_time = end_time + 1
    return data

def get_data(queries, start, end, host):
    data = {}
    for name, query in queries.items():
        print(f"Working on query for {name}...")
        data[name] = exec_query(query, start, end, host)

    columns = {}
    for m, containers in data.items():
        for c, info in containers.items():
            i = np.array(info)
            time = i[0:, 0]
            values = i[0:, 1]
            if len(columns) == 0:
                columns["time"] = time
            if (len(columns["time"]) < len(time)):
                columns["time"] = time
            columns[f"{c}_{m}"] = values
    return columns

def make_dict_list_equal(dict_list):
    l_min = float('inf')
    for key in dict_list:
        l_min = min(l_min, len(dict_list[key]))

    new_dict = {}
    for key, old_list in dict_list.items():
        new_list = old_list
        if len(old_list) > l_min:
            print(f"Discarding {len(old_list) - l_min} entries from the end of the column name {key}")
            new_list = old_list[:l_min]
        new_dict[key] = new_list
    return new_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collect data from Prometheus for sock-shop')

    parser.add_argument('--ip', type=str, required=True, help='The ip of vm/container running Prometheus')
    parser.add_argument('--start', type=int, required=True, help='The start time')
    parser.add_argument('--end', type=int, required=True, help='The end time')
    parser.add_argument('--name', type=str, default='data.csv', help='The name/path of the file')
    parser.add_argument('--append', action='store_true', help='Append to the file')

    args = parser.parse_args()
    ip = args.ip
    start = args.start
    end = args.end
    name = args.name
    append = args.append

    host = HOST_TEMPLATE.format(ip)
    df = pd.DataFrame(make_dict_list_equal(get_data(QUERIES, start, end, host)))
    if append:
        df.to_csv(name, index=False, mode='a', header=False)
    else:
        df.to_csv(name, index=False)

    print(f"The timeseries data is saved in file name {name}!")
    print(f"Total number of records are {len(df)}")
