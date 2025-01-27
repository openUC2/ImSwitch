# ensure flake8 compliance
try:
    c.NotebookApp.tornado_settings = {
        'headers': {
            'Content-Security-Policy': "frame-ancestors *"
        }
    }
    c.NotebookApp.ip = '0.0.0.0' # listen on all IPs 
except NameError:
    pass
