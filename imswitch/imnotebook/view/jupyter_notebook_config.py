# ensure flake8 compliance
try:
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
    c.NotebookApp.disable_check_xsrf = True  # Allow POST requests from iframes
    c.ServerApp.token = ''
    c.ServerApp.password = ''
    c.ServerApp.disable_check_xsrf = True  # Allow POST requests from iframes (JupyterLab)
except NameError:
    pass
