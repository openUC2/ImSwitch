# ensure flake8 compliance
try:
    c.NotebookApp.tornado_settings = {
        'headers': {
            'Content-Security-Policy': "frame-ancestors *"
        }
    }
except NameError:
    pass
