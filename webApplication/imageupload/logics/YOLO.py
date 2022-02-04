import torch
# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')


def OBJ_DETECTOR(img_path=None):
    img = img_path
    results = model(img)
    #results.save(save_dir='E:')
    df = results.pandas().xyxy[0]
    #obj_dict = df.to_dict(orient='dict')
    return df
