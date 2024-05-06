import cv2 
import torch 
import matplotlib.pyplot as plt 
from enum import Enum 
import os 
import numpy as np 

class ModelType(Enum): 
    DPT_LARGE = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    DPT_Hybrid = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    MIDAS_SMALL = "MiDaS_small" # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

class Midas(): 
    def __init__(self,modelType:ModelType=ModelType.DPT_LARGE): 
        self.midas = torch.hub.load("isl-org/MiDaS", modelType.value)
        self.modelType = modelType

    def useCUDA(self):
        if torch.cuda.is_available():
            print('Using CUDA')
            self.device = torch.device("cuda") 
        else: 
            print('Using CPU')
            self.device = torch.device("cpu")
        self.midas.to(self.device)
        self.midas.eval()

    def transform(self): 
        print('Transform')
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if self.modelType.value == "DPT_Large" or self.modelType.value == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform
    
    def predict(self,frame): 
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img).to(self.device)
        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depthMap = prediction.cpu().numpy()
        depthMap = cv2.normalize(depthMap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depthMap = cv2.applyColorMap(depthMap, cv2.COLORMAP_INFERNO)
        return depthMap
    
    def livePredict(self): 
        print('Starting webcam (press q to quit, press c to save images)...')
        capObj = cv2.VideoCapture(0)
        while True: 
            ret,frame = capObj.read()
            depthMap = self.predict(frame)
            combined = np.hstack((frame, depthMap))
            cv2.putText(combined, "Press 'C' to save image", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.namedWindow('Combined', cv2.WINDOW_NORMAL)

            # Resize the window
            cv2.resizeWindow('Combined', 1000, 1000)
            cv2.imshow('Combined',combined)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.save_images(frame, depthMap)
                print("Images saved successfully!")
        capObj.release()
        cv2.destroyAllWindows()
        
    def save_images(self, img, depth_map, save_folder="saved_images"):
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        cv2.imwrite(os.path.join(save_folder, "image.jpg"), img)
        cv2.imwrite(os.path.join(save_folder, "depth_map.jpg"), depth_map)

def run(modelType:ModelType):
    midasObj = Midas(modelType)
    midasObj.useCUDA() 
    midasObj.transform() 
    midasObj.livePredict() 

if __name__ == '__main__':
    run(ModelType.DPT_LARGE)
