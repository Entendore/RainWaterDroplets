"""
Entry point for the application.
Usage:
    python app.py
"""

import sys
from PySide6.QtWidgets import QApplication
from main_window import MainWindow


def main():
    """Initialize and run the application."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())