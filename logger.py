import logging

class Logger():
    def __init__(self):
        self.logger = logging.getLogger("Logger")
        self.stdout_handler = logging.StreamHandler()
        self.logger.addHandler(self.stdout_handler)
        self.stdout_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        self.logger.setLevel(logging.INFO)

    def info(self, txt):
        self.logger.info(txt)

    def close(self):
        self.stdout_handler.close()