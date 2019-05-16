import sys

if __name__ == "__main__":
    sys.path.append('QtApp')
    from QtApp import AppMain as app
    sys.exit(app.runApplication())