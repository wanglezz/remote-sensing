import unittest
from unittest.mock import patch, MagicMock
from models.training.train import train, validate, load_config


class TestTrain(unittest.TestCase):
    @patch('models.training.train.YOLO')
    @patch('models.training.train.load_config')
    def test_train(self, mock_load_config, mock_yolo):
        # Mock the config and YOLO model
        mock_load_config.return_value = {
            'data': {'dataset_yaml': './data/datasets/DOTA/yolo_obb/dataset.yaml', 'dataset_name': 'DOTA'},
            'model': {'pretrained': True, 'architecture': 'yolov8n-obb.pt'},
            'train': {
                'imgsz': 640,
                'epochs': 100,
                'batch_size': 16,
                'optimizer': 'Adam',
                'lr0': 0.001,
                'lrf': 0.01,
                'momentum': 0.9,
                'weight_decay': 0.0005,
                'warmup_epochs': 3,
                'warmup_momentum': 0.8,
                'augment': {
                    'hsv_h': 0.015,
                    'hsv_s': 0.7,
                    'hsv_v': 0.4,
                    'scale': 0.5,
                    'translate': 0.1,
                    'mosaic': 1.0,
                    'mixup': 0.0,
                },
                'box': 0.05,
                'cls': 0.5,
                'dfl': 1.0,
                'amp': True,
                'patience': 100,
                'save_period': -1,
            },
            'checkpoint': {
                'save_dir': './runs/train',
                'project': 'exp',
                'name': 'yolov8n-obb',
                'exist_ok': False,
                'resume': False,
            }
        }
        mock_yolo.return_value = MagicMock()

        # Call the train function
        train(config_path='./configs/training_config.yaml')

        # Assert that YOLO is called with the correct arguments
        mock_yolo.assert_called_once_with('yolov8n-obb.pt')
        mock_yolo.return_value.train.assert_called_once_with(
            data='./data/datasets/DOTA/yolo_obb/dataset.yaml',
            imgsz=640,
            epochs=100,
            batch=16,
            optimizer='Adam',
            lr0=0.001,
            lrf=0.01,
            momentum=0.9,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            scale=0.5,
            translate=0.1,
            mosaic=1.0,
            mixup=0.0,
            box=0.05,
            cls=0.5,
            dfl=1.0,
            amp=True,
            patience=100,
            save_period=-1,
            project='./runs/train/exp',
            name='yolov8n-obb',
            exist_ok=False,
            verbose=True,
            save=True,
            save_txt=True,
            save_csv=True,
        )

    @patch('models.training.train.YOLO')
    @patch('models.training.train.load_config')
    def test_validate(self, mock_load_config, mock_yolo):
        # Mock the config and YOLO model
        mock_load_config.return_value = {
            'val': {
                'imgsz': 640,
                'batch_size': 16,
                'iou_thres_nms': 0.65,
                'conf_thres': 0.001,
                'save_json': False,
                'save_hybrid': False,
            },
            'data': {
                'dataset_yaml': './data/datasets/DOTA/yolo_obb/dataset.yaml'
            }
        }
        mock_yolo.return_value = MagicMock()
        mock_yolo.return_value.val.return_value = MagicMock(
            box=MagicMock(map50=0.8, map=0.75, mp=0.9, mr=0.85)
        )

        # Call the validate function
        result = validate(model_path='./runs/train/exp/weights/best.pt', config_path='./configs/training_config.yaml')

        # Assert the validation results
        self.assertEqual(result['mAP50'], 0.8)
        self.assertEqual(result['mAP50-95'], 0.75)
        self.assertEqual(result['precision'], 0.9)
        self.assertEqual(result['recall'], 0.85)

if __name__ == '__main__':
    unittest.main()