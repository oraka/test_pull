
from data_generator.object_detection_2d_data_generator import DataGenerator
import json,sys
def make_dataset(images_dirs,annotations_dirs,image_set_filenames,output_dataset_h5_filepath,classes,include_classes):
    """
    データセットを作成する。
    images_dirs,annotations_dirs,image_set_filenamesを複数指定するときは
    順番を入れ替えてはいけない。
    例：データセットA,Bを使ってmake_datasetを呼び出すときは
    make_dataset([A.image,B.image],[A.anno,B.anno],[A.set,B.set],...)と順番を合わせる必要があり、
    make_dataset([A.image,B.image],[B.anno,A.anno],[A.set,B.set],...)とすることに対応していない

    annotation_dirsで指定するannotationのxmlの形式は以下
    <annotation>
	<folder>VOC2012</folder>
	<filename>2007_000033.jpg</filename>
	<object>
		<name>aeroplane</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>9</xmin>
			<ymin>107</ymin>
			<xmax>499</xmax>
			<ymax>263</ymax>
		</bndbox>
	</object>
	<object>
        ...(省略)
	</object>
    </annotation>
    
    Parameters
    ----------
    images_dirs : list[str, ...]
        画像が格納されたディレクトリのパスのリスト

    annotations_dirs : list[str, ...]
        アノテーションが格納されたディレクトリのリスト

    image_set_filenames : list[str, ...]
        image_dirsの中でdatasetに含める画像名のリスト

    output_dataset_h5_filepath : str
        データセットをh5形式で保存するときのファイルパス

    classes : str or list[str, ...]
        annotationに含まれるクラスのリスト
        学習に使わないクラスも含める必要がある
        例　VOC0712のリスト : ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat',
            'chair', 'cow', 'diningtable', 'dog',
            'horse', 'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor']

    include_classes : str or list[str, ...]
        "all"の場合、指定したデータセットに含まれるクラスをすべて含む。
        listで属性名を指定した場合、指定した属性の枠だけ抽出してデータセットを作成する。


    """
    
    #空のデータセットを作成
    dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)

    #データセット読み込み
    dataset.parse_xml(images_dirs=images_dirs,
                            image_set_filenames=image_set_filenames,
                            annotations_dirs=annotations_dirs,
                            classes=classes,
                            include_classes=include_classes,
                            exclude_truncated=False,
                            exclude_difficult=False,
                            ret=False)
    #h5ファイル生成
    dataset.create_hdf5_dataset(file_path=output_dataset_h5_filepath,
                                    resize=False,
                                    variable_image_size=True,
                                    verbose=True)

if __name__ == "__main__":
    if not len(sys.argv) == 5:
        sys.exit()

    image_dirs = sys.argv[1]
    annotation_dirs = sys.argv[2]
    image_set_filenames = sys.argv[3]
    output_dataset_h5_filepath= sys.argv[4]
    classes_json= sys.argv[5]
    include_classes_json = sys.argv[6]
    
    with open(classes_json) as f:
        classes = json.load(f)
    with open(include_classes_json) as f:
        include_classes = json.load(f)

    make_dataset([image_dirs],[annotation_dirs],[image_set_filenames],output_dataset_h5_filepath,classes,include_classes)