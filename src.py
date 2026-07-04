import os
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_name: str = 'yolov8n.pt'):
        """Initializes the YOLO model."""
        self.model = YOLO(model_name)

    def load_and_predict(self, image_path: str):
        """Runs the YOLO algorithm on a single image path."""
        results = self.model(image_path, verbose=False)
        return results[0]

    def evaluate_prediction(self, result) -> tuple:
        """
        Parses YOLO results to find the object detected with the highest confidence.
        Returns a tuple: (Class_Name: str, Confidence: float)
        """
        highest_conf = 0.0
        top_class_name = "None"

        if result.boxes and len(result.boxes) > 0:
            for box in result.boxes:
                conf = float(box.conf[0])
                
                if conf > highest_conf:
                    highest_conf = conf
                    class_id = int(box.cls[0])
                    top_class_name = self.model.names[class_id]

        return top_class_name, round(highest_conf, 4)

    def scan_folder(self, folder_path: str) -> list:
        """Loops through a folder, processes images, and populates a results list."""
        results_list = []
        
        valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')

        if not os.path.exists(folder_path):
            print(f"Error: Folder '{folder_path}' not found.")
            return results_list

        for filename in os.listdir(folder_path):
            if filename.lower().endswith(valid_extensions):
                image_path = os.path.join(folder_path, filename)
                
                raw_result = self.load_and_predict(image_path)
                
                prediction, confidence = self.evaluate_prediction(raw_result)
                
                results_list.append({
                    "image": filename,
                    "object_prediction": prediction,
                    "confidence": confidence
                })
                
        return results_list
    

class BirdClassifier:
    def __init__(self, target_label: str = "bird"):
        """Initializes the classifier with the target label to isolate."""
        self.target_label = target_label

    def clean_single_result(self, result_item: dict) -> dict:
        """
        Checks a single result dictionary. 
        If the detected object isn't the target_label, recodes it to 'None' and 0 confidence.
        """
        # Create a shallow copy to avoid mutating the original input list data unexpectedly
        cleaned_item = result_item.copy()
        current_label = cleaned_item.get("object_prediction")

        if current_label != self.target_label:
            cleaned_item["object_prediction"] = "None"
            cleaned_item["confidence"] = 0.0
            
        return cleaned_item

    def process_results(self, raw_results: list) -> list:
        """Loops through the list of outputs and applies the filtering logic."""
        processed_list = []
        
        for item in raw_results:
            cleaned = self.clean_single_result(item)
            processed_list.append(cleaned)
            
        return processed_list


# --- Execution Example ---
if __name__ == "__main__":
    folder_to_scan = "./images" 
    
    detector = ObjectDetector()
    analysis_results = detector.scan_folder(folder_to_scan)
    
    print("\n--- Detection Results ---")
    for result in analysis_results:
        print(result)

    classifier = BirdClassifier()
    final_results = classifier.process_results(analysis_results)

    print("--- Final Filtered Results ---")
    for final_item in final_results:
        print(final_item) 