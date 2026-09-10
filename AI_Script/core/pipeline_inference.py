import os
from AI_Script.models.factory_model import ModelFactory
from AI_Script.preprocess.factory_preprocess import PreprocessorFactory
from AI_Script.postprocess.factory_postprocess import PostprocessorFactory
from AI_Script.core.utils import check_file
import cv2
from datetime import datetime

class Pipeline_Inference:
    def __init__(self, config):
        self.config = config
        if not self.config.get("output_path") or not str(self.config.get("output_path")).strip():
            raise ValueError("'output_path' is required in config (no default).")
        self.output_path = os.path.abspath(str(self.config.get("output_path")).strip())
        os.makedirs(self.output_path, exist_ok=True)
        self.model_name = str(self.config.get("model_name"))
        # Display flag - default False for headless (only save final video)
        display_val = self.config.get("display", self.config.get("display_option", False))
        self.display = bool(display_val) if isinstance(display_val, bool) else str(display_val).lower() in ("true", "1", "yes")

        # pre-process -> AI inference -> post-process
        self.preprocessor = PreprocessorFactory.create(config=self.config)
        self.model = ModelFactory.create(config=self.config)
        self.postprocessor = PostprocessorFactory.create(config=self.config)

    def _process_single_item(self, item_source):
        preprocessed_data = self.preprocessor(item_source)
        model_output = self.model(preprocessed_data)
        return model_output

    def run(self, input_source):
        input_type = check_file(input_source)
        final_results = []

        # --- Case 1: single image ---
        if input_type in ['image_path', 'npy_path', 'numpy_array']:
            print("Detected single input. Processing...")
            # Step 1: pre-process -> ai-inference
            result_1 = self._process_single_item(input_source)
            # Step 2: post-process
            result_2 = self.postprocessor(result_1, input_source)
            final_results.append(result_2)

        # --- Case 2: folder images ---
        elif input_type == 'folder_path':
            print(f"Detected batch input (folder). Processing each item...")
            try:
                image_files = sorted(
                    [f for f in os.listdir(input_source) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                if not image_files:
                    print(f"Warning: No images found in folder {input_source}")
                    return []
            except FileNotFoundError:
                print(f"Error: Folder not found at {input_source}")
                return []

            for filename in image_files:
                item_path = os.path.join(input_source, filename)
                try:
                    dict_format = {"image_path": item_path, "result": None}
                    # Step 1: pre-process -> ai-inference
                    result_1 = self._process_single_item(item_path)
                    # Step 2: post-process
                    result_2 = self.postprocessor(result_1, item_path)
                    dict_format["result"] = result_2
                    final_results.append(dict_format)
                except Exception as e:
                    print(f"    ! Failed to process {filename}. Error: {e}")

        # --- Case 3: video ---
        elif input_type == 'video_path':
            print(f"Detected video input. Processing...")
            # Open video
            cap = cv2.VideoCapture(input_source)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Video FPS = {fps}")
            print(f"Width video = {width}")
            print(f"Height video = {height}")

            # Initialize video writer
            output_path = os.path.join(self.output_path, f"{self.model_name} video_detection {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.mkv")
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'FFV1')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # Loop video
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # AI model
                # Step 1: pre-process -> ai-inference
                result_1 = self._process_single_item(frame)
                # Step 2: post-process
                result_2 = self.postprocessor(result_1, frame, False)

                # Display frame (optional, default False for headless)
                if self.display:
                    cv2.imshow('ByteTrack Original Implementation', result_2)

                # Save frame (always)
                if output_path:
                    out.write(result_2)

                # out loop if push 'q' (only when display is enabled)
                if self.display and (cv2.waitKey(1) & 0xFF == ord('q')):
                    break

            # Cleanup
            cap.release()
            if output_path:
                out.release()
            if self.display:
                cv2.destroyAllWindows()

            # DONE
            print(f"Video are saved in {output_path}")

        else:
            raise ValueError(f"Unsupported input type: {input_type}")


    def __call__(self, input_source):
        return self.run(input_source)