import logging

# Define ANSI color codes
class Colors:
    BLACK = "\033[0;30m"
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    BROWN = "\033[0;33m"
    BLUE = "\033[0;34m"
    PURPLE = "\033[0;35m"
    CYAN = "\033[0;36m"
    LIGHT_GRAY = "\033[0;37m"
    DARK_GRAY = "\033[1;30m"
    LIGHT_RED = "\033[1;31m"
    LIGHT_GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    LIGHT_BLUE = "\033[1;34m"
    LIGHT_PURPLE = "\033[1;35m"
    LIGHT_PURPLE2 = "\033[94m"
    LIGHT_CYAN = "\033[1;36m"
    LIGHT_WHITE = "\033[1;37m"
    BOLD = "\033[1m"
    FAINT = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    NEGATIVE = "\033[7m"
    CROSSED = "\033[9m"
    END = "\033[0m"
    DARK_ORANGE = '\033[38;5;208m'

class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""
    suffix = (f"({Colors.BLUE}%(pathname)s{Colors.END}:"
              f"{Colors.DARK_ORANGE}%(lineno)d{Colors.END}:"
              f"{Colors.CYAN}%(funcName)s(){Colors.END})")

    format_dict = {
        logging.DEBUG:    f"{Colors.LIGHT_PURPLE2}DEBUG:    %(message)s{Colors.END} " + suffix,
        logging.INFO:     f"{Colors.CYAN}INFO:     %(message)s{Colors.END} " + suffix,
        logging.WARNING:  f"{Colors.DARK_ORANGE}WARNING:  %(message)s{Colors.END} " + suffix,
        logging.ERROR:    f"{Colors.RED}ERROR:    %(message)s{Colors.END} " + suffix,
        logging.CRITICAL: f"{Colors.RED}CRITICAL: %(message)s{Colors.END} " + suffix
    }

    def format(self, record):
        log_fmt = self.format_dict.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
    
def get_logger_with_level(level):
    log = logging.getLogger()
    log.setLevel(level)

    console_stream_handler = logging.StreamHandler()
    console_stream_handler.setLevel(level)
    console_stream_handler.setFormatter(CustomFormatter())

    log.addHandler(console_stream_handler)
    return log