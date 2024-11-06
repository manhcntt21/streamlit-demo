```python
    import os
    from pathlib import Path
    
    url = "https://github.com/serengil/deepface_models/releases/download/v1.0/retinaface.h5"
    home = str(os.getenv("DEEPFACE_HOME", default=str(Path.home())))

    exact_file = home + "/.deepface/weights/retinaface.h5"
```
download model from `url` and save to `exact_file` (or install retina-face from source and edit source code).
