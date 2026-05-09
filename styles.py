"""
Application theme and stylesheet definitions.
"""

APP_STYLESHEET = """
QMainWindow {
    background-color: #1A1D23;
}

#ControlPanel {
    background-color: #20242B;
    border-right: 1px solid #2E333D;
}

QLabel {
    color: #E0E0E0;
    font-family: 'Segoe UI', sans-serif;
}

#Title {
    font-size: 28px;
    font-weight: bold;
    color: #00BCD4;
}

#Subtitle {
    font-size: 12px;
    color: #757575;
    margin-bottom: 10px;
}

QGroupBox {
    font-weight: bold;
    font-size: 13px;
    color: #B0BEC5;
    border: 1px solid #2E333D;
    border-radius: 8px;
    margin-top: 10px;
    padding-top: 10px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
}

QSlider::groove:horizontal {
    border: 1px solid #2E333D;
    height: 6px;
    background: #12151A;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #00BCD4;
    border: 1px solid #00BCD4;
    width: 16px;
    margin: -5px 0;
    border-radius: 9px;
}

QPushButton {
    background-color: #2E333D;
    color: #FFFFFF;
    border: none;
    padding: 12px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #3A3F4B;
}

QPushButton:pressed {
    background-color: #12151A;
}

#RecordBtn {
    background-color: #D32F2F;
}

#RecordBtn:hover {
    background-color: #F44336;
}

QComboBox, QCheckBox {
    background-color: #12151A;
    color: white;
    border: 1px solid #2E333D;
    border-radius: 4px;
    padding: 5px;
}

QCheckBox::indicator {
    width: 13px;
    height: 13px;
}

#Footer {
    color: #546E7A;
    font-size: 11px;
}
"""

RECORDING_ACTIVE_STYLE = """
QPushButton {
    background-color: #C62828;
}
"""