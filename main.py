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
    
    # Anythin firewall related
    
     sudo firewall-cmd --add-port=8001/tcp --permanent
     sudo firewall-cmd --add-port=3232/tcp --permanent # for esp ota
     sudo firewall-cmd --add-port=3333/tcp --permanent # for esp ota
     sudo firewall-cmd --reload
     sudo firewall-cmd --list-ports | grep 8001

     # kill -9 $(lsof -ti:8001)

    '''
    # DON'T CHANGE THIS!!!!
    # This has to be maintained for DOCKER!
    main(is_headless=True)
