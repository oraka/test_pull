set classes_json=
set include_classes_json=

set image_dirs=
set annotation_dirs=
set image_set_filenames=
set output_dataset_h5_filepath=
python src\make_dataset.py %image_dirs% %annotation_dirs% %image_set_filenames% %output_dataset_h5_filepath% %classes_json% %include_classes_json


set image_dirs=
set annotation_dirs=
set image_set_filenames=
set output_dataset_h5_filepath=
python src\make_dataset.py %image_dirs% %annotation_dirs% %image_set_filenames% %output_dataset_h5_filepath% %classes_json% %include_classes_json