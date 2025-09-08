# ensure flake8 compliance
try:
    # Legacy notebook app configuration
    c.NotebookApp.tornado_settings = {
        'headers': {
            'Content-Security-Policy': "frame-ancestors *",
                 "Access-Control-Allow-Origin": "*",
                 "Access-Control-Allow-Headers": "Content-Type",
                 "Access-Control-Allow-Methods": "GET, POST, OPTIONS"
        }
    }
    c.NotebookApp.ip = '0.0.0.0' # listen on all IPs
    c.NotebookApp.allow_credentials = True
    c.NotebookApp.allow_origin = '*'
    c.NotebookApp.allow_remote_access = True
except NameError:
    pass

# Modern Jupyter Server configuration
try:
    c.ServerApp.tornado_settings = {
        'headers': {
            'Content-Security-Policy': "frame-ancestors *",
                 "Access-Control-Allow-Origin": "*",
                 "Access-Control-Allow-Headers": "Content-Type",
                 "Access-Control-Allow-Methods": "GET, POST, OPTIONS"
        }
    }
    c.ServerApp.ip = '0.0.0.0' # listen on all IPs
    c.ServerApp.allow_credentials = True
    c.ServerApp.allow_origin = '*'
    c.ServerApp.allow_remote_access = True
except NameError:
    pass
