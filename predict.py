"""Predict one new image without any ground-truth input."""
import argparse
from object_counting.io import ROOT, load_params, load_scene, read_image, resolve, new_run, digest
from object_counting.pipeline import count_image
from object_counting.visualization import save_prediction
from object_counting.experiments import touches_test, check_freeze, log_test_run


def predict(image_path, scene_id, config='configs/baseline.toml'):
    image_path=resolve(image_path)
    # A renamed copy of a test image is still a test image; same freeze rule as run_eval.
    if image_path.is_file() and touches_test([dict(image_id=image_path.stem,sha256=digest(image_path))]):
        log_test_run(ROOT,event='single_image_prediction',image=str(image_path),config=str(config),**check_freeze(config))
    params, _ = load_params(config)
    scene, background, roi = load_scene(scene_id)
    image = read_image(image_path)
    if scene.get('background_status') not in ('reviewed','legacy_offline_median'):
        print('Lưu ý: nền cảnh này còn cần nhóm rà soát.')
    prediction = count_image(image, background, params, roi, scene.get('perspective'))
    path = new_run('predictions')
    save_prediction(path, image, prediction, dict(image=str(image_path), scene_id=scene_id,
                                                background_status=scene.get('background_status')))
    return prediction, path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',required=True); parser.add_argument('--scene',required=True)
    parser.add_argument('--config',default='configs/baseline.toml')
    args=parser.parse_args()
    try:
        result,path=predict(args.image,args.scene,args.config)
    except (ValueError,OSError,KeyError) as error:
        parser.exit(2,str(error)+'\n')
    print(f'Dự đoán: {result.count} | Kết quả: {path}')


if __name__=='__main__':
    main()
