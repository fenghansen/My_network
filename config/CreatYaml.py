import yaml


# manage your network config here
config = {}

# generate yaml file
with open('./config.yaml', 'w', encoding='utf-8') as f:
    yaml.dump(config, f)