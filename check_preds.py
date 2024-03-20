import cv2
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box
from torchvision import transforms

def run_inference(weights_path, source_path, conf_thres=0.25, iou_thres=0.45, img_size=640, device='cpu'):
    # Load the model
    model = attempt_load(weights_path, map_location=device)
    model.eval()

    # Set device
    device = torch.device(device)
    model.to(device)

    # Get class names
    names = model.module.names if hasattr(model, 'module') else model.names

    # Resize transformation
    resize = transforms.Resize((img_size, img_size))
    to_tensor = transforms.ToTensor()

    while True:
        if isinstance(source_path, str):
            if source_path.endswith(('.jpg', '.png', '.jpeg')):
                # Image
                img = cv2.imread(source_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_tensor = to_tensor(img)
                img_tensor = resize(img_tensor)
                img_tensor = img_tensor.unsqueeze(0)
            else:
                # Video
                cap = cv2.VideoCapture(source_path)
                ret, frame = cap.read()
                if not ret:
                    break
                frame_tensor = to_tensor(frame)
                frame_tensor = resize(frame_tensor)
                frame_tensor = frame_tensor.unsqueeze(0)
        elif isinstance(source_path, int):
            # Webcam
            cap = cv2.VideoCapture(source_path)
            ret, frame = cap.read()
            if not ret:
                break
            frame_tensor = to_tensor(frame)
            frame_tensor = resize(frame_tensor)
            frame_tensor = frame_tensor.unsqueeze(0)

        # Inference
        with torch.no_grad():
            pred = model(img_tensor if isinstance(source_path, str) and source_path.endswith(('.jpg', '.png', '.jpeg')) else frame_tensor, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        # Process detections
        for det in pred:
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img_tensor.shape[2:] if isinstance(source_path, str) and source_path.endswith(('.jpg', '.png', '.jpeg')) else frame_tensor.shape[2:], det[:, :4], img.shape[:2] if isinstance(source_path, str) and source_path.endswith(('.jpg', '.png', '.jpeg')) else frame.shape[:2]).round()

                # Print results
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, img if isinstance(source_path, str) and source_path.endswith(('.jpg', '.png', '.jpeg')) else frame, label=label, color=(0, 255, 0), line_thickness=2)

        # Display the result
        if isinstance(source_path, str) and source_path.endswith(('.jpg', '.png', '.jpeg')):
            cv2.imshow('Output', img)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        else:
            cv2.imshow('Output', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Clean up
    if isinstance(source_path, str) and source_path.endswith(('.jpg', '.png', '.jpeg')):
        cv2.destroyAllWindows()
    else:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    weights_path = '/Users/maniksinghsarmaal/Downloads/yolov7/runs/train/exp7/weights/best.pt'  # Replace with the path to your trained weights
    source_path = 0  # Replace with the path to your image, video, or webcam (0 for default webcam)
    run_inference(weights_path, source_path)