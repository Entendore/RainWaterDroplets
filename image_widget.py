"""
Custom widget for displaying simulation output with aspect-ratio scaling.
"""

from PySide6.QtWidgets import QLabel
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap


class ImageWidget(QLabel):
    """QLabel subclass that maintains aspect ratio when displaying images."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 400)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scaled_pixmap = None
        self._original_pixmap = None

    def set_image(self, qimage):
        self._original_pixmap = QPixmap.fromImage(qimage)
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self):
        if self._original_pixmap:
            self._scaled_pixmap = self._original_pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.setPixmap(self._scaled_pixmap)

    def resizeEvent(self, event):
        self._update_scaled_pixmap()
        super().resizeEvent(event)

    @property
    def original_pixmap(self):
        return self._original_pixmap