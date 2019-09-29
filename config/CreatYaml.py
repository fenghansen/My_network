import yaml


# manage your network config here
config = {'img_height': 256, 'img_width': 256, 'img_channels': 3, 'p_height': 256,
          'p_width': 256, 'p_channels': 18, 'batch_size': 6}

# generate yaml file
with open('./config.yaml', 'w', encoding='utf-8') as f:
    yaml.dump(config, f)