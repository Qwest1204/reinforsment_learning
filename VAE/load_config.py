import json

def get_config_for(identific):
    with open("config.json", "r") as file:
        data = json.load(file)

    return data[identific]