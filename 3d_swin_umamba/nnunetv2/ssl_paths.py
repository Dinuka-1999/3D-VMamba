import os
join = os.path.join


base = join(os.sep.join(__file__.split(os.sep)[:-3]), 'data') 
ssl_raw = join(base, 'ssl_raw') # os.environ.get('nnUNet_raw')
ssl_preprocessed = join(base, 'ssl_preprocessed') # os.environ.get('nnUNet_preprocessed')

if ssl_raw is None:
    print("ssl_raw is not defined and SSL can only be used on data for which preprocessed files "
          "are already present on your system. SSL cannot be used for experiment planning and preprocessing like "
          "this. If this is not intended, please read documentation/setting_up_paths.md for information on how to set "
          "this up properly.")

if ssl_preprocessed is None:
    print("ssl_preprocessed is not defined and SSL can not be used for preprocessing "
          "or training. If this is not intended, please read documentation/setting_up_paths.md for information on how "
          "to set this up.")