Scrypted custom car make/model classifier

Use the URL of config.json as the Custom Model URL in the Scrypted CoreML Object Detection plugin. Keep the model.mlpackage directory and the three files in config.json's files array at the same URL path.

This classifier takes a single vehicle crop. It does not supply bounding boxes, so retain Scrypted's normal vehicle object detector upstream.
