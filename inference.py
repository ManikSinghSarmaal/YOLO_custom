#RUN THE CODE BELOW ON TERMINAL
# 1. webcam -> python detect.py --weights /Users/maniksinghsarmaal/Downloads/yolov7/runs/train/exp7/weights/best.pt --source 0 --conf 0.75 --img-size 640 --view-img --device
import cv2
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box

def run_inference(weights_path, source_path, conf_thres=0.25, iou_thres=0.45, img_size=320, device='cpu'):
    # Load the model
    model = attempt_load(weights_path, map_location=device)
    model.eval()

    # Set device
    device = torch.device(device)
    model.to(device)

    # Load the source (image or video)
    if source_path.endswith(('.jpg', '.png', '.jpeg')):
        # Image
        img = cv2.imread(source_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        # Video
        cap = cv2.VideoCapture(source_path)

    # Get class names
    names = model.module.names if hasattr(model, 'module') else model.names

    while True:
        if source_path.endswith(('.jpg', '.png', '.jpeg')):
            # Image
            img_tensor = torch.from_numpy(img).to(device)
            img_tensor = img_tensor.float() / 255.0  # Normalize to [0, 1]
            if img_tensor.ndimension() == 3:
                img_tensor = img_tensor.unsqueeze(0)

            # Inference
            with torch.no_grad():
                pred = model(img_tensor, augment=False)[0]

            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres)

            # Process detections
            for det in pred:
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], img.shape[:2]).round()

                    # Print results
                    for *xyxy, conf, cls in reversed(det):
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, img, label=label, color=(0, 255, 0), line_thickness=2)

            # Display the result
            cv2.imshow('Output', img)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        else:
            # Video
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess the frame
            frame_tensor = torch.from_numpy(frame).to(device)
            frame_tensor = frame_tensor.float() / 255.0  # Normalize to [0, 1]
            if frame_tensor.ndimension() == 3:
                frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)

            # Inference
            with torch.no_grad():
                pred = model(frame_tensor, augment=False)[0]

            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres)

            # Process detections
            for det in pred:
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(frame_tensor.shape[2:], det[:, :4], frame.shape[:2]).round()

                    # Print results
                    for *xyxy, conf, cls in reversed(det):
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, frame, label=label, color=(0, 255, 0), line_thickness=2)

            # Display the result
            cv2.imshow('Output', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Clean up
    if source_path.endswith(('.jpg', '.png', '.jpeg')):
        cv2.destroyAllWindows()
    else:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    weights_path = '/Users/maniksinghsarmaal/Downloads/yolov7/runs/train/exp7/weights/last.pt'  # Replace with the path to your trained weights
    source_path = '/Users/maniksinghsarmaal/Downloads/z.mp4'  # Replace with the path to your image or video
    run_inference(weights_path, source_path)