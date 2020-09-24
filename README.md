# Litening

Project aiming at target recognition using Litening targeting pod images within DCS combat simulator.

## Usage

Currently system is designed to classify 4 classes:
- T-55
- BTR-80
- BMP-1
- T-72B

and localize 2 classes:
- BMP-1
- BTR-80

## Requirements

Project is written using **Python 3.5** and uses following libraries:
- Keras 2.1.5
- matplotlib 2.2.2
- numpy 1.17.0
- pytest 3.5.0
- scikit-learn 0.19.1
- tensorflow 1.5.0

## Testing

All the tests are located in **test** directory and can be launched using PyTest in command line:
```python -m pytest test```

## Development

Project is currently put on hold, but will be resumed with following development plan:
- refactoring code and project structure
- improving classification and localization task
- adding more data
- implementing multiple object detection with localization
