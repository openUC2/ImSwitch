if __name__ == '__main__':
    from imswitch.__main__ import main
    '''
    To start imswitch in headless with a remote config file, you can add additional arguments:
    main(is_headless=True, 
         default_config="/Users/bene/ImSwitchConfig/imcontrol_setups/example_virtual_microscope.json", 
         http_port=8001, ssl=True, data_folder="/Users/bene/Downloads")
    - is_headless: True or False
    - default_config: path to the config file
    - http_port: port number
    - ssl: True or False
    - data_folder: path to the data folder
    example:
    main(is_headless=True, data_folder="/Users/bene/Downloads")
    
     sudo firewall-cmd --zone=public --add-port=8001/tcp; sudo firewall-cmd --zone=nm-shared --add-port=8001/tcp
     sudo firewall-cmd --zone=public --add-port=8002/tcp; sudo firewall-cmd --zone=nm-shared --add-port=8002/tcp
    '''
    # DON'T CHANGE THIS!!!!
    # This has to be maintained for DOCKER!
    main(is_headless=True)
